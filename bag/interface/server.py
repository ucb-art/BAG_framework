# -*- coding: utf-8 -*-

"""This class defines SkillOceanServer, a server that handles skill/ocean requests.

The SkillOceanServer listens for skill/ocean requests from bag.  Skill commands will
be forwarded to Virtuoso for execution, and Ocean simulation requests will be handled
by starting an Ocean subprocess.  It also provides utility for bag to query simulation
progress and allows parallel simulation.

Client-side communication:

the client will always send a request object, which is a python dictionary.
This script processes the request and sends the appropriate commands to
Virtuoso.

Virtuoso side communication:

To ensure this process receive all the data from Virtuoso properly, Virtuoso
will print a single line of integer indicating the number of bytes to read.
Then, virtuoso will print out exactly that many bytes of data, followed by
a newline (to flush the standard input).  This script handles that protcol
and will strip the newline before sending result back to client.
"""

import traceback

import bag.io


def _object_to_skill_file_helper(py_obj, file_obj):
    """Recursive helper function for object_to_skill_file

    Parameters
    ----------
    py_obj : any
        the object to convert.
    file_obj : file
        the file object to write to.  Must be created with bag.io
        package so that encodings are handled correctly.
    """
    # fix potential raw bytes
    py_obj = bag.io.fix_string(py_obj)
    if isinstance(py_obj, str):
        # string
        file_obj.write(py_obj)
    elif isinstance(py_obj, float):
        # prepend type flag
        file_obj.write('#float {:f}'.format(py_obj))
    elif isinstance(py_obj, bool):
        bool_val = 1 if py_obj else 0
        file_obj.write('#bool {:d}'.format(bool_val))
    elif isinstance(py_obj, int):
        # prepend type flag
        file_obj.write('#int {:d}'.format(py_obj))
    elif isinstance(py_obj, list) or isinstance(py_obj, tuple):
        # a list of other objects.
        file_obj.write('#list\n')
        for val in py_obj:
            _object_to_skill_file_helper(val, file_obj)
            file_obj.write('\n')
        file_obj.write('#end')
    elif isinstance(py_obj, dict):
        # disembodied property lists
        file_obj.write('#prop_list\n')
        for key, val in py_obj.items():
            file_obj.write('{}\n'.format(key))
            _object_to_skill_file_helper(val, file_obj)
            file_obj.write('\n')
        file_obj.write('#end')
    else:
        raise Exception('Unsupported python data type: %s' % type(py_obj))


def object_to_skill_file(py_obj, file_obj):
    """Write the given python object to a file readable by Skill.

    Write a Python object to file that can be parsed into equivalent
    skill object by Virtuoso.  Currently only strings, lists, and dictionaries
    are supported.

    Parameters
    ----------
    py_obj : any
        the object to convert.
    file_obj : file
        the file object to write to.  Must be created with bag.io
        package so that encodings are handled correctly.
    """
    _object_to_skill_file_helper(py_obj, file_obj)
    file_obj.write('\n')


bag_proc_prompt = 'BAG_PROMPT>>> '


class SkillServer(object):
    """A server that handles skill commands.

    This server is started and ran by virtuoso.  It listens for commands from bag
    from a ZMQ socket, then pass the command to virtuoso.  It then gather the result
    and send it back to bag.

    Parameters
    ----------
    router : :class:`bag.interface.ZMQRouter`
        the :class:`~bag.interface.ZMQRouter` object used for socket communication.
    virt_in : file
        the virtuoso input file.  Must be created with bag.io
        package so that encodings are handled correctly.
    virt_out : file
        the virtuoso output file.  Must be created with bag.io
        package so that encodings are handled correctly.
    tmpdir : str or None
        if given, will save all temporary files to this folder.
    """

    def __init__(self, router, virt_in, virt_out, tmpdir=None):
        """Create a new SkillOceanServer instance.
        """
        self.handler = router
        self.virt_in = virt_in
        self.virt_out = virt_out

        # create a directory for all temporary files
        self.dtmp = bag.io.make_temp_dir('skillTmp', parent_dir=tmpdir)

    def run(self):
        """Starts this server.
        """
        while not self.handler.is_closed():
            # check if socket received message
            if self.handler.poll_for_read(5):
                req = self.handler.recv_obj()
                if isinstance(req, dict) and 'type' in req:
                    if req['type'] == 'exit':
                        self.close()
                    elif req['type'] == 'skill':
                        expr, out_file = self.process_skill_request(req)
                        if expr is not None:
                            # send expression to virtuoso
                            self.send_skill(expr)
                            msg = self.recv_skill()
                            self.process_skill_result(msg, out_file)
                    else:
                        msg = '*Error* bag server error: bag request:\n%s' % str(req)
                        self.handler.send_obj(dict(type='error', data=msg))
                else:
                    msg = '*Error* bag server error: bag request:\n%s' % str(req)
                    self.handler.send_obj(dict(type='error', data=msg))

    def send_skill(self, expr):
        """Sends expr to virtuoso for evaluation.

        Parameters
        ----------
        expr : string
            the skill expression.
        """
        self.virt_in.write(expr)
        self.virt_in.flush()

    def recv_skill(self):
        """Receive response from virtuoso"""
        num_bytes = int(self.virt_out.readline())
        msg = self.virt_out.read(num_bytes)
        if msg[-1] == '\n':
            msg = msg[:-1]
        return msg

    def close(self):
        """Close this server."""
        self.handler.close()

    def process_skill_request(self, request):
        """Process the given skill request.

        Based on the given request object, returns the skill expression
        to be evaluated by Virtuoso.  This method creates temporary
        files for long input arguments and long output.

        Parameters
        ----------
        request : dict
            the request object.

        Returns
        -------
        expr : str or None
            expression to be evaluated by Virtuoso.  If None, an error occurred and
            nothing needs to be evaluated
        out_file : str or None
            if not None, the result will be written to this file.
        """
        try:
            expr = request['expr']
            input_files = request['input_files'] or {}
            out_file = request['out_file']
        except KeyError as e:
            msg = '*Error* bag server error: %s' % str(e)
            self.handler.send_obj(dict(type='error', data=msg))
            return None, None

        fname_dict = {}
        # write input parameters to files
        for key, val in input_files.items():
            with bag.io.open_temp(prefix=key, delete=False, dir=self.dtmp) as file_obj:
                fname_dict[key] = '"%s"' % file_obj.name
                # noinspection PyBroadException
                try:
                    object_to_skill_file(val, file_obj)
                except Exception:
                    stack_trace = traceback.format_exc()
                    msg = '*Error* bag server error: \n%s' % stack_trace
                    self.handler.send_obj(dict(type='error', data=msg))
                    return None, None

        # generate output file
        if out_file:
            with bag.io.open_temp(prefix=out_file, delete=False, dir=self.dtmp) as file_obj:
                fname_dict[out_file] = '"%s"' % file_obj.name
                out_file = file_obj.name

        # fill in parameters to expression
        expr = expr.format(**fname_dict)
        return expr, out_file

    def process_skill_result(self, msg, out_file=None):
        """Process the given skill output, then send result to socket.

        Parameters
        ----------
        msg : str
            skill expression evaluation output.
        out_file : str or None
            if not None, read result from this file.
        """
        # read file if needed, and only if there are no errors.
        if msg.startswith('*Error*'):
            # an error occurred, forward error message directly
            self.handler.send_obj(dict(type='error', data=msg))
        elif out_file:
            # read result from file.
            try:
                msg = bag.io.read_file(out_file)
                data = dict(type='str', data=msg)
            except IOError:
                stack_trace = traceback.format_exc()
                msg = '*Error* error reading file:\n%s' % stack_trace
                data = dict(type='error', data=msg)
            self.handler.send_obj(data)
        else:
            # return output from virtuoso directly
            self.handler.send_obj(dict(type='str', data=msg))
