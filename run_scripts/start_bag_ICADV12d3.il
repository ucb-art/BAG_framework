/*  Note:

Due to licensing reasons, this skill script is missing the function 
CCSinvokeCdfCallbacks() from Cadence solution 11018344, which executes 
CDF parameters callback from skill.

If you do not need to instantiate a pcell instance, this method
is not needed.

Eric Chang, Mar 2, 2017.

*/


;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;  Virtuoso Database operations functions  ;;
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
; reads a skill data structure from file
procedure( parse_data_from_file( fname "t" )
    let( (p ans)
        unless( p = infile( fname )
            error("Cannot open file %s" fname)
        )
        ans = parse_data_from_file_helper(p)
        close( p )
        ans
    )
)

; recursive helper for parse_data_from_file
procedure( parse_data_from_file_helper( p )
    let( (line item ans finish key)
        gets( line p )
        ; remove newline
        line = substring(line 1 strlen(line) - 1)
        ; printf("read line: %s\n" line)
        cond(
            (line == "#list"
            ; parse a list
            ans = tconc(nil 0)
            while( nequal(item = parse_data_from_file_helper(p) "#end")
                tconc(ans item)
            )
            ; printf("returning list ")
            ; print(cdar(ans))
            ; printf("\n")
            cdar(ans)
            )
            (line == "#prop_list"
            ; parse a disembodied property list
            ans = ncons(nil)
            finish = nil
            while( !finish
                key = parse_data_from_file_helper(p)
                if( key == "#end" then
                    finish = 't
                else
                    item = parse_data_from_file_helper(p)
                    putprop(ans item key)
                )
            )
            ans
            )
            ; parse a float
            (strncmp( line "#float" 6 ) == 0
                cdfParseFloatString(cadr(parseString(line)))
            )
            ; parse an int
            (strncmp( line "#int" 4 ) == 0
                atoi(cadr(parseString(line)))
            )
            ; parse a boolean
            (strncmp( line "#bool" 5 ) == 0
                if( atoi(cadr(parseString(line))) == 1 then
                    't
                else
                    nil
                )
            )
            ; parse a string token or #end
            ('t
                ; printf("returning str %s\n" line)
                line
            )
        )
    )
)

; return a list of cells in the given library.
procedure( get_cells_in_library( lib_name "t" )
    let( ( lib_obj ans )
        if( lib_obj = ddGetObj(lib_name nil nil nil nil "r") then
            ans = ddGetObjChildren(lib_obj)~>name
            ddReleaseObj(lib_obj)
        else
            ; library does not exist, return empty list
            ans = '()
        )
        ans
    )
)

; return a list of cells in the given library.
procedure( get_cells_in_library_file( lib_name fname "tt" )
    let( ( p )
        p = outfile( fname "w" )
        foreach( cell get_cells_in_library(lib_name)
            fprintf(p "%s\n" cell)
        )
        close(p)
    )
)

; Returns the directory corresponding to the given library.
procedure( get_lib_directory(lib_name "t")
    let( ( lib_obj ans )
        if( lib_obj = ddGetObj(lib_name nil nil nil nil "r") then
            ans = lib_obj~>readPath
            ddReleaseObj(lib_obj)
        else
            ; library does not exist, return empty list
            ans = ""
        )
        ans
    )
)

; Parse the netlist of the given cellview.
; Works on schematic and veriloga.
procedure( parse_cad_sch(lib_name cell_name file_name "ttt")
    let( (cv cell_type p indent direction term_names tb_list tb_match
          inst_lib_name inst_cell_name inst_cnt)
        indent = ""
        cell_type = "schematic"
        unless( cv = dbOpenCellViewByType( lib_name cell_name "schematic" nil "r" )
            cell_type = "veriloga"
            unless( cv = dbOpenCellViewByType( lib_name cell_name "veriloga" nil "r" )
                error( "Cannot find schematic or veriloga view of cell %s__%s" lib_name cell_name )
            )
        )
        p = outfile( file_name "w" )

        ; print cellview information
        printf( "*INFO* Writing cell %s__%s (%s) netlist to %s\n" lib_name cell_name cell_type file_name )
        fprintf( p "%slib_name: %s\n" indent lib_name )
        fprintf( p "%scell_name: %s\n" indent cell_name )

        ; print pins
        fprintf( p "%spins: [ " indent )
        if( cell_type == "veriloga" then
           term_names = reverse(cv~>terminals~>name)
        else
           term_names = cv~>terminals~>name
        )
        ; add quotes around pin names to escape array pins
        term_names = mapcar( lambda( (x) sprintf(nil "\"%s\"" x) ) term_names )
        fprintf( p "%s ]\n" buildString(term_names ", "))

        ; print instances
        if( not(cv~>instances) then
            fprintf( p "%sinstances: {}\n" indent )
        else
            inst_cnt = 0
            fprintf( p "%sinstances:\n" indent )
            foreach( inst cv~>instances
                inst_cnt++
                ; print entry for instance
                indent = "  "
                fprintf( p "%s%s:\n" indent inst~>name )
                ; print instance master information.
                indent = "    "
                fprintf( p "%slib_name: %s\n" indent inst~>libName )
                fprintf( p "%scell_name: %s\n" indent inst~>cellName )
                ; print instance terminal information
                if( !(inst~>instTerms) then
                    fprintf( p "%sinstpins: {}\n" indent )
                else
                    fprintf( p "%sinstpins:\n" indent )
                    foreach( inst_term inst~>instTerms
                        unless( direction = inst_term~>direction
                            direction = ""
                        )
                        indent = "      "
                        fprintf( p "%s%s:\n" indent inst_term~>name )
                        indent = "        "
                        fprintf( p "%sdirection: %s\n" indent direction )
                        fprintf( p "%snet_name: \"%s\"\n" indent inst_term~>net~>name )
                        fprintf( p "%snum_bits: %d\n" indent inst_term~>numBits )
                    )
                )
            )
            when(inst_cnt == 0
                fprintf( p "  {}\n" )
            )
        )

        ; close resources
        close(p)
        dbClose(cv)
    )
)

; Delete a cellview if it exists.  Currently used to delete old calibre file.
procedure( delete_cellview(lib_name cell_name view_name "ttt")
    let( (obj)
        obj = ddGetObj(lib_name cell_name view_name)
        if( obj then
            ddDeleteObj(obj)
        else
            't
        )
    )
)

; Parse the structure of the given cellview.
; Works on layout.
procedure( parse_cad_layout(lib_name cell_name file_name "ttt")
    let( (cv cell_type p indent rect_cnt label_cnt inst_cnt)

        indent = ""
        cell_type = "layout"
        unless( cv = dbOpenCellViewByType( lib_name cell_name cell_type nil "r" )
            error( "Cannot find layout view of cell %s__%s" lib_name cell_name )
        )
        p = outfile( file_name "w" )

        ; print cellview information
        printf( "*INFO* Writing cell %s__%s (%s) netlist to %s\n" lib_name cell_name cell_type file_name )
        fprintf( p "%slib_name: %s\n" indent lib_name )
        fprintf( p "%scell_name: %s\n" indent cell_name )

        ; print rects
        if( not(cv~>shapes) then
            fprintf( p "%srects: {}\n" indent )
        else
            rect_cnt = 0
            fprintf( p "%srects:\n" indent )
            foreach( shape cv~>shapes
                if( (shape~>objType == "rect") then
                    rect_cnt++ 
                    ; print entry for rect
                    indent = "  "
                    fprintf( p "%s%d:\n" indent rect_cnt )
                    ; print rect master information.
                    indent = "    "
                    fprintf( p "%slayer: %s %s\n" indent nthelem(1 shape~>lpp) nthelem(2 shape~>lpp))
                    fprintf( p "%sbBox: [[%f, %f], [%f, %f]]\n" indent 
                                 nthelem(1 nthelem(1 shape~>bBox)) nthelem(2 nthelem(1 shape~>bBox)) 
                                 nthelem(1 nthelem(2 shape~>bBox)) nthelem(2 nthelem(2 shape~>bBox)) 
                    );fprintf
                )
            );if
            if((rect_cnt == 0) then
                fprintf( p "  {}\n" )
           );if
        )

        ; print labels
        indent = ""
        if( not(cv~>shapes) then
            fprintf( p "%slabels: {}\n" indent )
        else
            label_cnt = 0
            fprintf( p "%slabels:\n" indent )
            foreach( shape cv~>shapes
                if( (shape~>objType == "label") then
                    label_cnt++ 
                    ; print entry for label
                    indent = "  "
                    fprintf( p "%s%d:\n" indent label_cnt )
                    ; print label master information.
                    indent = "    "
                    fprintf( p "%slabel: %s\n" indent shape~>theLabel )
                    fprintf( p "%slayer: %s %s\n" indent nthelem(1 shape~>lpp) nthelem(2 shape~>lpp))
                    fprintf( p "%sxy: [%f, %f]\n" indent nthelem(1 shape~>xy) nthelem(2 shape~>xy))
                )
                if( (shape~>objType == "textDisplay") then ;some labels are instantiated as text displays
                    label_cnt++
                    ; print entry for label
                    indent = "  "
                    fprintf( p "%s%d:\n" indent label_cnt )
                    ; print label master information.
                    indent = "    "
                    fprintf( p "%slabel: %s\n" indent shape~>owner~>name )
                    fprintf( p "%slayer: %s %s\n" indent nthelem(1 shape~>lpp) nthelem(2 shape~>lpp))
                    fprintf( p "%sxy: [%f, %f]\n" indent nthelem(1 shape~>xy) nthelem(2 shape~>xy))
                )
            );if
            if((label_cnt == 0) then
                fprintf( p "  {}\n" )
           );if
        )
        
        ; print instances
        indent = ""
        if( not(cv~>instances) then
            fprintf( p "%sinstances: {}\n" indent )
        else
            inst_cnt = 0
            fprintf( p "%sinstances:\n" indent )
            foreach( inst cv~>instances
                inst_cnt++
                ; print entry for instance
                indent = "  "
                fprintf( p "%s%s:\n" indent inst~>name )
                ; print instance master information.
                indent = "    "
                fprintf( p "%slib_name: %s\n" indent inst~>libName )
                fprintf( p "%scell_name: %s\n" indent inst~>cellName )
                fprintf( p "%sxy: [%f, %f]\n" indent nthelem(1 inst~>xy) nthelem(2 inst~>xy))
                if( (inst~>objType == "mosaic") then
                    fprintf( p "%scols: %d\n" indent inst~>columns)
                    fprintf( p "%srows: %d\n" indent inst~>rows)
                    fprintf( p "%ssp_cols: %f\n" indent inst~>uX)
                    fprintf( p "%ssp_rows: %f\n" indent inst~>uY)
                    fprintf( p "%srotation: %s\n" indent car(inst~>tileArray))
                    else
                    fprintf( p "%srotation: %s\n" indent inst~>orient)
                );if
            )
            when(inst_cnt == 0
                fprintf( p "  {}\n" )
            )
        )
        
        ; close resources
        close(p)
        dbClose(cv)
    )
)

; get a list of cells containing in the specficied library
procedure( get_cell_list(lib_name file_name "tt")
    let( (lib cellname p)
        lib=ddGetObj(lib_name)
        p = outfile( file_name "w" )
        fprintf( p "%s: [" lib_name)
        foreach( cellname lib~>cells~>name
            fprintf( p "%s, " cellname)
        );foreach
        fprintf( p "] \n" )
        ; close resources
        close(p)
    );let
)

; if library with lib_name does not exists, create a new
; library with that name.  Otherwise, if erase is true,
; remove all cells in that library.  Returns the library
; database object.
procedure( create_or_erase_library(lib_name tech_lib lib_path erase "tttg")
    let( (lib_obj)
        if( lib_obj = ddGetObj(lib_name nil nil nil nil "r") then
            when( erase
                ; delete all cells in the library
                foreach( cell lib_obj~>cells
                    unless( ddDeleteObj(cell)
                        error("cannot delete cell %s in library %s\n" cell~>name lib_name)
                    )
                )
            )
            ddReleaseObj(lib_obj)
            't
        else
            ; create library if not exist
            when( and(lib_path (lib_path != "."))
                lib_path = strcat(lib_path "/" lib_name)
            )
            lib_obj = ddCreateLib(lib_name lib_path)
            ; attach technology file
            techBindTechFile(lib_obj tech_lib)
            ; close library
            ddReleaseObj(lib_obj)
            't
        )
    )
)

; copy all template cells to the given library.
; template list is a list of three-element lists with the format
; '("master_lib_name" "master_cell_name" "target_cell_name")
; any existing cellviews will be overwritten.
procedure( copy_templates_to_library(lib_name template_list "tl")
    let( (current remaining src_gdm targ_gdm table master_lib master_cell target_cell key cnt
          empty_spec targ_lib_obj test_cv)

        current = template_list
        remaining = '()
        empty_spec = gdmCreateSpecList()
        targ_lib_obj = ddGetObj(lib_name nil nil nil nil "r")

        ; ccpCopy cannot copy the same cell to multiple different cells.
        ; because of this, we need to copy a set of unique cells at a time,
        ; hence the while loop.
        while( current
            ; Create GDMSpecList used to copy all cells
            src_gdm = gdmCreateSpecList()
            targ_gdm = gdmCreateSpecList()
            ; table to keep track of seen cells.
            table = makeTable("mytable" 0)
            ; Populate GDMSpecList
            foreach( template_info current
                master_lib = car(template_info)
                master_cell = cadr(template_info)
                target_cell = caddr(template_info)

                ; check if we copied this cell on this iteration yet
                key = list(master_lib master_cell)
                if( table[key] == 1 then
                    ; wait for the next iteration
                    remaining = cons(template_info remaining)
                else
                    ; purge target cellview if exist
                    when( targ_lib_obj
                        test_cv = dbFindOpenCellView(targ_lib_obj target_cell "schematic")
                        when( test_cv
                            dbPurge(test_cv)
                        )
                        test_cv = dbFindOpenCellView(targ_lib_obj target_cell "symbol")
                        when( test_cv
                            dbPurge(test_cv)
                        )
                        ; hard remove adexl state if it exists
                        test_cv = ddGetObj(lib_name target_cell "adexl")
                        when( test_cv
                            ddDeleteObj(test_cv)
                        )
                    )
                    gdmAddSpecToSpecList(gdmCreateSpec(master_lib master_cell nil nil "CDBA") src_gdm)
                    gdmAddSpecToSpecList(gdmCreateSpec(lib_name target_cell nil nil "CDBA") targ_gdm)
                    table[key] = 1
                )
            )
            ; Perform copy
            ccpCopy(src_gdm targ_gdm 't 'CCP_EXPAND_COMANAGED nil nil "" "" 'CCP_UPDATE_FROM_LIBLIST empty_spec)

            ; set current and remaining
            current = remaining
            remaining = '()

            ; debug printing
            ; printstruct(table)
        )
    )
    't
)

; returns a unique terminal name in the given cellview.
; name_base is the suffix of the returned terminal name.
procedure( get_unique_term_name( cvid name_base "gt")
    let( (cnt new_term_name)
        cnt = 1
        sprintf( new_term_name "temp%d_%s" cnt name_base )
        while( dbFindTermByName(cvid new_term_name)
            cnt = cnt + 1
            sprintf( new_term_name "temp%d_%s" cnt name_base )
        )
        new_term_name
    )
)

; helper method to open pin master
procedure( open_pin_master(cvid pin_cv_info)
    let( (pin_master mpin_lib mpin_cell mpin_view)
        mpin_lib = car(pin_cv_info)
        mpin_cell = cadr(pin_cv_info)
        mpin_view = caddr(pin_cv_info)
        unless( pin_master = dbOpenCellViewByType( mpin_lib mpin_cell mpin_view nil "r" )
            dbClose(cvid)
            error( "Cannot find pin master cellview: %s__%s (%s)" mpin_lib mpin_cell mpin_view)
        )
        pin_master
    )
)

; update pins of a schematic
; cvid is the opened cellview id of the schematic.  It must be in append mode.
; pin_map is a list of two-element lists of old pin names and new pin names, respectively.
; ipin, opin, and iopin are lists of three strings for input/output/inout pins, respectively.
; first element is the pin master library, second element is the pin mater cell, and third element
; is the pin master cellview.
procedure( update_schematic_pin(cvid pin_map new_pins ipin opin iopin "glllll")
    let( (snap_dist cur_term_name new_term_name term pin pin_orient pin_location pin_direction
          temp_new_term_name pin_master ipin_master opin_master iopin_master
          pin_xy_info npin_xl npin_yl npin_xr npin_yr npin_name npin_type)

        snap_dist = schGetEnv("schSnapSpacing")

        ; open pin masters
        ipin_master = open_pin_master(cvid ipin)
        opin_master = open_pin_master(cvid opin)
        iopin_master = open_pin_master(cvid iopin)
        pin_master = nil

        ; get new pin locations before any pin addition/substraction.
        pin_xy_info = get_new_pin_locations(cvid snap_dist)

        ; rename or remove pins
        foreach( p pin_map
            cur_term_name = car(p)
            new_term_name = cadr(p)
            ; printf("%s %s\n" cur_term_name new_term_name)
            when(cur_term_name != new_term_name
                unless( term = dbFindTermByName(cvid cur_term_name)
                    dbClose(cvid)
                    dbClose(ipin_master)
                    dbClose(opin_master)
                    dbClose(iopin_master)
                    error( "Terminal %s not found." cur_term_name )
                )
                when( term~>pinCount != 1
                    dbClose(cvid)
                    dbClose(ipin_master)
                    dbClose(opin_master)
                    dbClose(iopin_master)
                    error( "Terminal %s does not have exactly one pin." cur_term_name)
                )
                pin = car(term~>pins)

                if( strlen(new_term_name) != 0 then
                    ; rename pin
                    pin_orient = pin~>fig~>orient
                    pin_location = pin~>fig~>xy
                    pin_direction = term~>direction

                    ; create new pin figure
                    cond( ( pin_direction == "input" pin_master = ipin_master)
                          ( pin_direction == "output" pin_master = opin_master)
                          ( 't pin_master = iopin_master)
                    )

                    ; delete pin
                    unless( dbDeleteObject(pin~>fig)
                        dbClose(cvid)
                        dbClose(ipin_master)
                        dbClose(opin_master)
                        dbClose(iopin_master)
                        error( "Cannot delete pin for terminal %s" cur_term_name )
                    )

                    ; create a temporary terminal with a unique name so we can change the number of bits without getting an error
                    temp_new_term_name = get_unique_term_name(cvid new_term_name)
                    schCreatePin(cvid pin_master temp_new_term_name pin_direction nil pin_location "R0" )

                    ; now rename the new terminal
                    new_term = dbFindTermByName(cvid temp_new_term_name )
                    new_term~>name = new_term_name
                else
                    ; remove pin
                    dbDeleteObject(pin~>fig)
                )
            )
        )

        ; add new pins
        when( new_pins
            ; get location for new pins
            npin_xl = xCoord(car(pin_xy_info))
            npin_yl = yCoord(car(pin_xy_info)) - 2 * snap_dist
            npin_xr = xCoord(cadr(pin_xy_info))
            npin_yr = yCoord(cadr(pin_xy_info)) - 2 * snap_dist
            foreach( npin_info new_pins
                npin_name = car(npin_info)
                npin_type = cadr(npin_info)

                ; verify that this pin does not exist yet
                when(dbFindTermByName(cvid npin_name)
                    dbClose(cvid)
                    dbClose(ipin_master)
                    dbClose(opin_master)
                    dbClose(iopin_master)
                    error( "Terminal %s already exists" npin_name)
                )

                ; get pin location based on pin type
                cond( ( npin_type == "input" pin_master = ipin_master pin_location = npin_xl:npin_yl npin_yl = npin_yl - 2 * snap_dist)
                      ( npin_type == "output" pin_master = opin_master pin_location = npin_xr:npin_yr npin_yr = npin_yr - 2 * snap_dist)
                      ( 't pin_master = iopin_master pin_location = npin_xl:npin_yl npin_yl = npin_yl - 2 * snap_dist)
                )
                ; create pin
                schCreatePin(cvid pin_master npin_name npin_type nil pin_location "R0")
            )
        )

        dbClose(ipin_master)
        dbClose(opin_master)
        dbClose(iopin_master)
    )
)

; find X and Y coordinates to insert new symbol pins
procedure( get_new_pin_locations(cvid snap_dist)
    let( (pin bbox pin_x pin_y xl xr yl yr)
        ; find the left-most/right-most pin X coordinates, and find the lowst
        ; Y coordinate of the left-most/right-most pins
        xl = nil
        xr = nil
        yl = nil
        yr = nil
        foreach( term cvid->terminals
            when( term~>pinCount != 1
                dbClose(cvid)
                error( "Terminal %s does not have exactly one pin" term~>name)
            )
            pin = car(term~>pins)
            bbox = pin~>fig~>bBox
            pin_x = round2((xCoord(car(bbox)) + xCoord(cadr(bbox))) / 2.0 / snap_dist)
            pin_y = round2((yCoord(car(bbox)) + yCoord(cadr(bbox))) / 2.0 / snap_dist)
            if( xl == nil then
                xl = pin_x
                xr = pin_x
                yl = pin_y
                yr = pin_y
            else
                cond( (pin_x < xl xl = pin_x yl = pin_y)
                      (pin_x == xl yl = min(yl pin_y)))
                cond( (pin_x > xr xr = pin_x yr = pin_y)
                      (pin_x == xr yr = min(yr pin_y)))
            )
        )
        when(xl == nil
            ; default values if schematic has no terminals
            ; this usually means you have a testbench schematic
            xl = 0
            yl = 0
            xr = 10
            yr = 0
        )
        list((xl * snap_dist):(yl * snap_dist) (xr * snap_dist):(yr * snap_dist))
    )
)

; update pins of a symbol
; pin_map is a list of two-element lists, first element is old pin name, second element is new pin name.
; sympin is a 3-element list of strings. first element is the pin master library,
; second element is the pin mater cell, and third element is the pin master cellview.
; simulators is a list of simulator names for which termOrder should be updated.
; Usually simulators = '("auLvs" "auCdl" "spectre" "hspiceD")
procedure( update_symbol_pin(lib_name cell_name pin_map new_pins sympin simulators "ttllll")
    let( (snap_dist cvid pin_master cur_term_name new_term_name term pin bbox pin_x pin_y pin_location pin_direction
          label_location label_rel_location temp_new_term_name new_term new_port_order cell_obj bc
          mpin_lib mpin_cell mpin_view pin_xy_info npin_xl npin_yl npin_xr npin_yr npin_name npin_type
          modified_pins)

        snap_dist = schGetEnv("schSnapSpacing")
        modified_pins = nil
        mpin_lib = car(sympin)
        mpin_cell = cadr(sympin)
        mpin_view = caddr(sympin)
        unless( pin_master = dbOpenCellViewByType(mpin_lib mpin_cell mpin_view nil "r")
            error("Cannot open symbol pin cellview %s__%s (%s)." mpin_lib mpin_cell mpin_view)
        )
        unless( cvid = dbOpenCellViewByType(lib_name cell_name "symbol" nil "a")
            dbClose(pin_master)
            error("Cannot open cellview %s__%s (symbol)." lib_name cell_name)
        )

        ; get new pin locations before any pin addition/substraction.
        pin_xy_info = get_new_pin_locations(cvid snap_dist)

        ; modify existing pins
        new_port_order = tconc(nil "")
        foreach( p pin_map
            cur_term_name = car(p)
            new_term_name = cadr(p)
            new_port_order = tconc(new_port_order new_term_name)
            when( cur_term_name != new_term_name
                modified_pins = 't
                ; printf("%s %s\n" cur_term_name new_term_name)
                unless( term = dbFindTermByName(cvid cur_term_name)
                    dbClose(pin_master)
                    dbReopen(cvid, "r")
                    dbClose(cvid)
                    error( "Terminal %s not found." cur_term_name )
                )
                when( term~>pinCount != 1
                    dbClose(pin_master)
                    dbReopen(cvid, "r")
                    dbClose(cvid)
                    error( "Terminal %s does not have exactly one pin." cur_term_name)
                )
                pin = car(term~>pins)

                if( strlen(new_term_name) != 0 then
                    ; rename pin
                    bbox = pin~>fig~>bBox
                    pin_x = (xCoord(car(bbox)) + xCoord(cadr(bbox))) / 2.0
                    pin_y = (yCoord(car(bbox)) + yCoord(cadr(bbox))) / 2.0
                    pin_location = round2(pin_x / snap_dist) * snap_dist:round2(pin_y / snap_dist) * snap_dist
                    pin_direction = term~>direction

                    ; change label
                    prog( (label_orientation label_font label_font_size label_type label_text)
                        foreach( label pin~>fig~>children
                            when( label~>objType == "label"
                                label_location = label~>xy
                                label_orientation = label~>orient
                                label_rel_location = label~>justify
                                label_font = label~>font
                                label_font_size = label~>height
                                label_type = label~>labelType
                                label_text = label~>theLabel
                                when( label_text == cur_term_name
                                    schCreateSymbolLabel(cvid label_location "pin label" new_term_name label_rel_location
                                                         label_orientation label_font label_font_size label_type)
                                    return('t)
                                )
                            )
                        )
                        return(nil)
                    )

                    dbDeleteObject(pin~>fig)
                    dbDeleteObject(pin)

                    ;create a temporary terminal with a unique name so we can change the number of bits without getting an error
                    temp_new_term_name = get_unique_term_name(cvid new_term_name)
                    schCreateSymbolPin(cvid pin_master temp_new_term_name pin_direction pin_location "R0" )

                    new_term = dbFindTermByName(cvid temp_new_term_name )
                    dbDeleteObject(term)
                    new_term~>name = new_term_name
                else
                    ; remove pin
                    dbDeleteObject(pin~>fig)
                    dbDeleteObject(pin)
                    dbDeleteObject(term)
                )
            )
        )

        ; add new pins
        when( new_pins
            modified_pins = 't
            ; get location for new pins
            npin_xl = xCoord(car(pin_xy_info))
            npin_yl = yCoord(car(pin_xy_info)) - 2 * snap_dist
            npin_xr = xCoord(cadr(pin_xy_info))
            npin_yr = yCoord(cadr(pin_xy_info)) - 2 * snap_dist
            foreach( npin_info new_pins
                npin_name = car(npin_info)
                npin_type = cadr(npin_info)

                ; verify that this pin does not exist yet
                when(dbFindTermByName(cvid npin_name)
                    dbClose(pin_master)
                    dbReopen(cvid, "r")
                    dbClose(cvid)
                    error( "Terminal %s already exists" npin_name)
                )

                ; update pin order
                new_port_order = tconc(new_port_order npin_name)

                ; get pin location based on pin type
                if( equal(npin_type "output") then
                    label_location = npin_xr:npin_yr
                    label_rel_location = "lowerLeft"
                    npin_yr = npin_yr - 2 * snap_dist
                else
                    label_location = npin_xl:npin_yl
                    label_rel_location = "lowerRight"
                    npin_yl = npin_yl - 2 * snap_dist
                )

                ; create label and pin
                schCreateSymbolLabel(cvid label_location "pin label" npin_name label_rel_location
                                     "R0" "stick" snap_dist "normalLabel")
                schCreateSymbolPin(cvid pin_master npin_name npin_type label_location "R0")
            )
        )

        dbClose(pin_master)

        when( modified_pins
            ; update pin order
            new_port_order = cdar(new_port_order)
            schEditPinOrder(cvid new_port_order 't)
            dbSave(cvid)

            ; update termOrder for each simulators
            cell_obj = ddGetObj(lib_name cell_name nil nil nil "r")
            unless( bc = cdfGetBaseCellCDF(cell_obj)
                ddReleaseObj(cell_obj)
                dbReopen(cvid, "r")
                dbClose(cvid)
                error("Cannot find CDF parameters for %s__%s.  Delete generated cell and try again" lib_name cell_name)
            )
            foreach( simu simulators
                get(bc->simInfo simu)->termOrder = new_port_order
            )
            unless( cdfSaveCDF(bc)
                ddReleaseObj(cell_obj)
                dbReopen(cvid, "r")
                dbClose(cvid)
                error("Cannot save termOrder CDF for %s__%s." lib_name cell_name)
            )
            ddReleaseObj(cell_obj)
        )
        ; opening schematic will open all symbols inside that schematic.
        ; as the result, dbClose may not close this symbol view.  To get rid
        ; of edit lock, we use dbReopen so even if dbClose fails the edit lock
        ; will be gone.
        dbReopen(cvid, "r")
        dbClose(cvid)
    )
)

; record an association list from pin name to pin location in units of snap distances.
; the pin name is sorted alphabetically so we can use the equal function to test
; for equality.
procedure( get_instance_pin_info(inst "g")
    let( (snap_dist term_name pin_fig xval yval inst_term_xy ans)
        ans = nil
        snap_dist = schGetEnv("schSnapSpacing")
        foreach( term inst->master->terminals
            term_name = term~>name
            ; get terminal coordinate in symbol
            pin_fig = car(term~>pins)~>fig
            bbox = pin_fig~>bBox
            xval = (xCoord(car(bbox)) + xCoord(cadr(bbox))) / 2.0
            yval = (yCoord(car(bbox)) + yCoord(cadr(bbox))) / 2.0
            ; quantize to schematic snap spacing to avoid floating point rounding error.
            inst_term_xy = round2(xval / snap_dist):round2(yval / snap_dist)
            ans = cons(list(term_name inst_term_xy) ans)
        )
        sortcar(ans nil)
    )
)

; get all the wire objects connected to terminals of the given instance.
; we assume each terminal has exactly one pin with 1 wire connected, with a
; single label on the wire.  The wire doesn't connect to anything else.
; returns an association list from terminal name to a list of net name and wire figure object.
procedure( get_instance_terminal_wires(sch inst "gg")
    let( (snap_dist term_name pin_fig xval yval inst_term_xy net_name ans net_map)
        ans = nil
        net_map = nil
        snap_dist = schGetEnv("schSnapSpacing")
        foreach( inst_term inst~>instTerms
            term_name = inst_term~>name
            ; printf("terminal name: %s\n" term_name)
            when( inst_term~>term~>pinCount != 1
                dbClose(sch)
                error("Terminal %s must have exactly one pin." term_name)
            )
            unless( pin_fig = car(inst_term~>term~>pins)~>fig
                dbClose(sch)
                error("Cannot find pin figure for terminal %s" term_name)
            )
            ; get instance terminal coordinate in schematic
            bbox = dbTransformBBox(pin_fig~>bBox inst~>transform)
            ; printf("terminal pin fig bbox: %A\n" bbox)
            xval = xCoord(car(bbox)) + (xCoord(cadr(bbox)) - xCoord(car(bbox))) / 2.0
            yval = yCoord(car(bbox)) + (yCoord(cadr(bbox)) - yCoord(car(bbox))) / 2.0
            ; quantize to schematic snap spacing to avoid floating point rounding error.
            inst_term_xy = round2(xval / snap_dist) * snap_dist:round2(yval / snap_dist) * snap_dist
            net_name = inst_term~>net~>name
            net_map = cons(list(term_name net_name) net_map)
            ; printf("terminal pin x/y: %A\n" inst_term_xy)
            foreach( fig inst_term~>net~>figs
                points = fig~>points
                ; printf("figure points: %A\n" points)
                when( member(inst_term_xy points)
                    when( length(points) != 2
                        error("pin for terminal %s must be connected to a single wire with label" term_name)
                    )
                    ; printf("adding figure for terminal %s\n" term_name)
                    ans = cons(list(term_name fig) ans)
                )
            )
        )
        list(ans net_map)
    )
)

; Modify the instance terminal connections of the given instance.
; we assume each terminal to modify has at most 1 wire connected,
; if it exists, the wire connects to nothing else, and it has a label.
; In this way, this function just have to change the label text.
;
; if wire_list is not empty, then that means each terminal has exactly one
; wire connected.  This function will update the label on the wires according
; to term_mapping.
;
; if wire_list is empty, then that means no wires are connected to terminals.
; this function will attach labels directly to each terminal.  The labels are
; determined first from term_mapping, then from net_map
;
; sch is the schematic database object.  Must be opened in append/write mode.
; inst is the instance object to modify.
; term_mapping is a list of key-value pairs, where keys are old net names,
; and values are new net names.
procedure( modify_instance_terminal(sch inst wire_list net_map term_mapping "gglll")
    let( (snap_dist key_val old_name new_name fig points mid_point new_wire inst_term inst_pin
          bbox xval yval term_map_final db_term)
        ; get schematic snap distance spacing.
        snap_dist = schGetEnv("schSnapSpacing")
        if( wire_list then
            foreach( wire_info wire_list
                old_name = car(wire_info)
                when(key_val = assoc(old_name term_mapping)
                    new_name = cadr(key_val)
                    fig = cadr(wire_info)
                    points = fig~>points
                    mid_point = foreach(mapcar (c1 c2) car(points) cadr(points) (c1 + c2) / 2.0)
                    ; delete old wire, then add wire back with new label.
                    schDelete(fig)
                    new_wire = car(schCreateWire(sch "draw" "full" points snap_dist snap_dist 0))
                    schCreateWireLabel(sch new_wire mid_point new_name "lowerCenter" "R0" "stick" 0.0625 nil)
                )
            )
            't
        else
            ; combine net_map and term_mapping
            term_map_final = copy(term_mapping)
            foreach( net_info net_map
                old_name = car(net_info)
                unless( assoc(old_name term_map_final)
                    ; add net mapping only if it's not in term_mapping
                    term_map_final = cons(net_info term_map_final)
                )
            )
            foreach( net_info term_map_final
                old_name = car(net_info)
                new_name = cadr(net_info)

                when(db_term = dbFindTermByName(inst->master old_name)
                    ; only create terminal that's present in the current master
                    inst_term = dbCreateInstTerm(nil inst db_term)
                    inst_pin = car(inst_term~>term~>pins)~>fig
                    bbox = dbTransformBBox(inst_pin~>bBox inst~>transform)
                    xval = (xCoord(car(bbox)) + xCoord(cadr(bbox))) / 2.0
                    yval = (yCoord(car(bbox)) + yCoord(cadr(bbox))) / 2.0
                    xval = round2(xval / snap_dist) * snap_dist
                    yval = round2(yval / snap_dist) * snap_dist

                    new_wire = car(schCreateWire(sch "draw" "full" list(xval-snap_dist:yval-snap_dist xval:yval)
                                   snap_dist snap_dist 0))
                    schCreateWireLabel(sch new_wire xval:yval new_name "lowerCenter" "R0" "stick" 0.0625 nil)
                )
            )
            't
        )
    )
)

; Perform check-and-save on the given schematic database object, then close it.
procedure( check_and_save_schematic(sch "g")
    let( (errs)
        schSetEnv( "checkHierSave" 't)
        schSetEnv( "saveAction" "Save")
        errs = schCheckHier(sch "schematic symbol" "")
        foreach( ex errs
            warn( "%s__%s (%s) has %d errors." car(ex)~>lib~>name car(ex)~>cellName car(ex)~>viewName cadr(ex))
        )
        ; make sure all edit locks are gone by reopening in read mode
        dbReopen(sch, "r")
        dbClose(sch)
    )
)


; modify a schematic cell.  Used to convert copied template cells into concrete instantiation.
;
; inst_list is an association list of (inst_name, rinst_list) pairs.  Where:
;
; inst_name : name of the instance in the template cell.
; rinst_list : a list of rinsts, which are instances to replace the original instance by.
;              If this list is empty, the original instance should be deleted.  If the list
;              has more than one element, we should array the original instance.
;
; Each rinst is a disembodied property lists, with the properties:
;
; rinst->name : the name of this rinst.
; rinst->lib_name : the instance master library.
; rinst->cell_name : the instance master cell.
; rinst->params : an association list of the CDF params of this rinst.  The values are always string.
; rinst->term_mapping : an association list of the modified terminal connections of this rinst.
;                       if no connections are changed, this list should be empty.
;
; (You can read more about disembodied property lists and association list in the skill
; language user guide).
;
; For each instance, this function does the following:
; 1. Find the instance with the given name.
; 2. If rinst_list is nil, delete this instance.
; 3. If rinst_list has exactly one element:
;    i.   rename the instance name to rinst's name.
;    ii.  change the instance master of the instance.
;    iii. change the CDF parameters (this should only happen with BAG primitives).
;    iv.  change the port connections of this instance.
; 4. If rinst_list has more than one element, for each additional element,
;    copy the original instance and perform step 3 on that instance.
;
; This procedure allows one to delete or array any instances in the schematic template.
procedure( modify_schematic_content(sch_cv inst_list "gl")
    let( (inst_obj inst_name rinst_list rinst_len cur_inst wire_list net_map par_val xl xr transform
          snap_dist errmsg pin_info tmp_result)
        snap_dist = schGetEnv("schSnapSpacing")
        foreach( inst inst_list
            inst_name = car(inst)
            unless( inst_obj = dbFindAnyInstByName(sch_cv inst_name)
                dbClose(sch_cv)
                error( "Cannot find instance %s" inst_name )
            )
            rinst_list = cadr(inst)
            rinst_len = length(rinst_list)
            last_inst = nil
            if( rinst_len == 0 then
                ; no instances to replace by, delete.
                wire_list = car(get_instance_terminal_wires(sch_cv inst_obj))
                ; delete wires connected to instance
                foreach( wire_info wire_list
                    schDelete(cadr(wire_info))
                )
                ; delete instance
                dbDeleteObject(inst_obj)
            else
                cur_inst = nil
                pin_info = nil
                foreach( rinst rinst_list
                    if( !cur_inst then
                        cur_inst = inst_obj
                        ; printf("inst %s lib = %s, cell = %s\n" inst_name inst_obj->master->libName inst_obj->master->cellName)
                        tmp_result = get_instance_terminal_wires(sch_cv cur_inst)
                        net_map = cadr(tmp_result)
                        wire_list = car(tmp_result)
                        pin_info = get_instance_pin_info(cur_inst)
                        ; printf("%s wire_list: %A\n" inst_name wire_list)
                        ; figure out bounding box for potential future array
                        ; printf("instance %s bbox: %A\n" cur_inst~>name cur_inst~>bBox)
                        xl = xCoord(car(cur_inst~>bBox))
                        xr = xCoord(cadr(cur_inst~>bBox))
                        foreach( wire_info wire_list
                            ; printf("instance %s wire: %A %A\n" cur_inst~>name xCoord(car(cadr(wire_info)~>bBox)) xCoord(cadr(cadr(wire_info)~>bBox)))
                            xl = min(xl xCoord(car(cadr(wire_info)~>bBox)))
                            xr = max(xr xCoord(cadr(cadr(wire_info)~>bBox)))
                        )
                        transform = list(round2((xr - xl + snap_dist) / snap_dist) * snap_dist:0 "R0" 1.0)
                        ; printf("instance %s transform: %A\n" cur_inst~>name transform)
                    else
                        ; more than 1 rinst, copy cur_inst, do not copy wires
                        wire_list = nil
                        ; copy instance
                        cur_inst = dbCopyFig(cur_inst nil transform)
                    )
                    ; change instance name and master
                    when(cur_inst->name != rinst->name
                        cur_inst->name = rinst->name
                    )
                    schReplaceProperty(list(cur_inst) "master" sprintf(nil "%s %s %s" rinst->lib_name
                                                                       rinst->cell_name cur_inst->viewName))
                    ; set parameters
                    foreach( cdf_par cdfGetInstCDF(cur_inst)~>parameters
                        par_val = cadr(assoc(cdf_par->name rinst->params))
                        ; change CDF parameter value only if specified in given parameters
                        when( par_val != nil
                            cdf_par->value = par_val
                        )
                    )
                    when( wire_list
                        ; if wire_list is not empty, check that the pins match.  If so, keep wires around,
                        ; otherwise, delete wires
                        unless( equal(pin_info get_instance_pin_info(cur_inst))
                            ; delete wires connected to instance
                            foreach( wire_info wire_list
                                schDelete(cadr(wire_info))
                            )
                            wire_list = nil
                        )
                    )
                    ; modify connections, keeping old wires around
                    ; printf("instance %s wire_list: %A net_map: %A term_map: %A\n" cur_inst~>name wire_list net_map rinst->term_mapping)
                    modify_instance_terminal(sch_cv cur_inst wire_list net_map rinst->term_mapping)
                )
            )
        )
    )
)

; given a copied template cell, modify it to a concrete schematic.
procedure( convert_template_cells(lib_name cell_name pin_map new_pins inst_list sympin ipin opin iopin simulators)
    let( (sym_cv sch)
        ; update symbol view first.
        if( sym_cv = dbOpenCellViewByType(lib_name cell_name "symbol" nil "r") then
            printf("*INFO* Updating %s__%s symbol pins.\n" lib_name cell_name)
            update_symbol_pin(lib_name cell_name pin_map new_pins sympin simulators)
        else
            warn("Did not find symbol for %s__%s.  Skipping.  Is it testbench?" lib_name cell_name)
        )

        ; attempt to open schematic in append mode
        unless( sch = dbOpenCellViewByType(lib_name cell_name "schematic" nil "a")
            error("Cannot open %s__%s (schematic) in append mode." lib_name cell_name)
        )
        ; update schematic content
        printf("*INFO* Updating %s__%s instances and connections.\n" lib_name cell_name)
        modify_schematic_content(sch inst_list)
        ; update schematic pins
        printf("*INFO* Updating %s__%s schematic pins.\n" lib_name cell_name)
        update_schematic_pin(sch pin_map new_pins ipin opin iopin)
        check_and_save_schematic(sch)
    )
)

; create concrete schematics
procedure( create_concrete_schematic( lib_name tech_lib lib_path temp_file change_file
                                      sympin ipin opin iopin simulators copy "tttttlllllg" )
    let( (template_list change_list cell_name pin_map inst_list)
        printf("*INFO* Reading template and change list from file\n")
        template_list = parse_data_from_file( temp_file )
        change_list = parse_data_from_file( change_file )
        when( copy
            printf("*INFO* Creating library: %s\n" lib_name)
            create_or_erase_library( lib_name tech_lib lib_path nil )
            printf("*INFO* Copying templates to library: %s\n" lib_name)
            copy_templates_to_library( lib_name template_list )
        )
        foreach( change change_list
            cell_name = change->name
            pin_map = change->pin_map
            new_pins = change->new_pins
            inst_list = change->inst_list
            printf("*INFO* Updating cell %s__%s\n" lib_name cell_name)
            convert_template_cells( lib_name cell_name pin_map new_pins inst_list
                                    sympin ipin opin iopin simulators )
        )
        't
    )
)

; create a new layout view then instantiate a single pcell instance.
; this method also copy all the labels in the pcell top level.  In this way LVS/PEX will
; work correctly.
; params is a list of (variable_name type_string value) lists.
; pin_mapping is a list of (old_pin new_pin) lists.
procedure( create_layout_with_pcell(lib_name cell_name view_name inst_lib inst_cell params_f pin_mapping_f "ttttttt")
    let( (lay_cv inst_master inst inst_shapes label_location label_orientation label_lpp
          label_just label_font label_height label_type label_text params pin_mapping)
        unless( lay_cv = dbOpenCellViewByType(lib_name cell_name view_name "maskLayout" "w")
            error("Cannot open cellview %s__%s (%s)." lib_name cell_name view_name)
        )
        unless( inst_master = dbOpenCellViewByType(inst_lib inst_cell "layout" "maskLayout" "r")
            dbClose(lay_cv)
            error("Cannot open cellview %s__%s (layout)." inst_lib inst_cell)
        )

        params = parse_data_from_file(params_f)
        pin_mapping = parse_data_from_file(pin_mapping_f)

        inst = dbCreateParamInst(lay_cv inst_master "XTOP" '(0 0) "R0" 1 params)
        inst_shapes = inst~>master~>shapes

        foreach(shape inst_shapes
            when( shape->objType == "label"
                label_location = shape~>xy
                label_orientation = shape~>orient
                label_lpp = shape~>lpp
                label_just = shape~>justify
                label_font = shape~>font
                label_height = shape~>height
                label_type = shape~>labelType
                label_text = shape~>theLabel
                when( cadr(assoc(label_text pin_mapping))
                    label_text = cadr(assoc(label_text pin_mapping))
                )
                dbCreateLabel(lay_cv label_lpp label_location label_text label_just label_orientation label_font label_height )
            )
        )

        dbClose(inst_master)
        dbSave(lay_cv)
        dbClose(lay_cv)
    )
)

; helper for creating a path segment
procedure( create_path_seg_helper(cv lay p0 p1 width start_s end_s)
    let( (diag_ext info_list bext eext)
        if( and(car(p0) != car(p1) cadr(p0) != cadr(p1)) then
            diag_ext = width / 2
            width = width * sqrt(2)
        else
            diag_ext = width * sqrt(2) / 2
        )

        bext = 0
        eext = 0
        when( start_s == "round"
            bext = width / 2
            start_s = "custom"
        )
        when( end_s == "round"
            eext = width / 2
            end_s = "custom"
        )
        info_list = list(bext eext list(diag_ext diag_ext width/2 diag_ext diag_ext width/2))
        dbCreatePathSeg(cv lay p0 p1 width start_s end_s info_list)
    )
)


; helper for creating a path
procedure( create_path_helper( cv path )
    let( (lay width points estyle jstyle p0 p1 plen idx start_s end_s)
        lay = path->layer
        width = path->width
        points = path->points
        estyle = path->end_style
        jstyle = path->join_style
        p0 = nil
        plen = length(points)
        idx = 0
        foreach( cur_point points
            p1 = cur_point
            when( idx > 0
                if( idx == 1 then
                    start_s = estyle
                else
                    start_s = jstyle
                )
                if( idx == plen - 1 then
                    end_s = estyle
                else
                    end_s = jstyle
                )
                create_path_seg_helper(cv lay p0 p1 width start_s end_s)
            )
            p0 = p1
            idx = idx + 1
        )
    )
)


; helper for creating a single layout view
procedure( create_layout_helper( cv tech_file inst_list rect_list via_list pin_list path_list
                                 blockage_list boundary_list polygon_list "ggllllllll" )
    let( (inst_cv obj via_def via_enc1 via_enc2 enc1 enc2 off1 off2 via_params make_pin_rect
          pin_bb pin_w pin_h pin_xc pin_yc pin_orient label_h param_order orig_shape arr_dx arr_dy)

        ; create instances
        foreach( inst inst_list
            if( inst_cv = dbOpenCellViewByType( inst->lib inst->cell inst->view nil "r" ) then

                if( and( inst->num_rows==1 inst->num_cols==1) then
                    if( inst->params != nil then
                        ; create pcell instance
                        obj = dbCreateParamInst(cv inst_cv inst->name inst->loc inst->orient 1 inst->params)
                        ; execute parameter callbacks
                        when( obj
                            if( inst->param_order != nil then
                                param_order = inst->param_order
                            else
                                param_order = mapcar( lambda( (x) car(x) ) inst->params )
                            )
                            CCSinvokeCdfCallbacks(obj ?order param_order)
                        )
                    else
                        obj = dbCreateInst(cv inst_cv inst->name inst->loc inst->orient)
                    )
                else
                    if( inst->params != nil then
                        ; create pcell mosaic
                        obj = dbCreateParamSimpleMosaic(cv inst_cv inst->name inst->loc inst->orient
                                                        inst->num_rows inst->num_cols inst->sp_rows inst->sp_cols
                                                        inst->params)
                        ; execute parameter callbacks
                        when( obj
                            if( inst->param_order != nil then
                                param_order = inst->param_order
                            else
                                param_order = mapcar( lambda( (x) car(x) ) inst->params )
                            )
                            CCSinvokeCdfCallbacks(obj ?order param_order)
                        )
                    else
                        obj = dbCreateSimpleMosaic(cv inst_cv inst->name inst->loc inst->orient
                                                   inst->num_rows inst->num_cols inst->sp_rows inst->sp_cols)
                    )
                )
                unless( obj
                    warn("Error creating instance %s of %s__%s (%s).  Skipping." inst->name inst->lib inst->cell inst->view)
                )

            else
                warn("Cannot find instance %s__%s (%s).  Skipping." inst->lib inst->cell inst->view)
            )
        )

        ; create rectangles
        foreach( rect rect_list
            orig_shape = dbCreateRect(cv rect->layer rect->bbox)
            if( not(orig_shape) then
                warn("Error creating rectangle of layer %A.  Skipping." rect->layer)
            else
                when( rect->arr_nx != nil
                    for(icol 2 rect->arr_nx
                        arr_dx = rect->arr_spx * (icol - 1)
                        for(irow 1 rect->arr_ny
                            arr_dy = rect->arr_spy * (irow - 1)
                            dbCopyFig(orig_shape nil list(arr_dx:arr_dy "R0" 1))
                        )
                    )
                    for(irow 2 rect->arr_ny
                        arr_dy = rect->arr_spy * (irow - 1)
                        dbCopyFig(orig_shape nil list(0:arr_dy "R0" 1))
                    )
                )
            )
        )

        ; create paths
        foreach( path path_list
            create_path_helper(cv path)
        )

        ; create polygons
        foreach( poly polygon_list
            dbCreatePolygon(cv poly->layer poly->points)
        )

        ; create blockages
        foreach( block blockage_list
            if( block->btype == "placement" then
                dbCreateAreaBlockage(cv block->points)
            else
                dbCreateLayerBlockage(cv block->layer block->btype block->points)
            )
        )

        ; create boundaries
        foreach( bound boundary_list
            cond( (bound->btype == "PR"
                   dbCreatePRBoundary(cv bound->points))
                  (bound->btype == "snap"
                   dbCreateSnapBoundary(cv bound->points))
                  (bound->btype == "area"
                   dbCreateAreaBoundary(cv bound->points))
                  ('t
                   warn("Unknown boundary type %s.  Skipping." bound->btype))
            )
        )

        ; create vias
        foreach( via via_list
            if( via_def = techFindViaDefByName(tech_file via->id) then
                ; compute via parameter list
                via_enc1 = via->enc1
                via_enc2 = via->enc2
                enc1 = list( (car(via_enc1) + cadr(via_enc1)) / 2.0
                             (caddr(via_enc1) + cadr(cddr(via_enc1))) / 2.0 )
                enc2 = list( (car(via_enc2) + cadr(via_enc2)) / 2.0
                             (caddr(via_enc2) + cadr(cddr(via_enc2))) / 2.0 )
                off1 = list( (cadr(via_enc1) - car(via_enc1)) / 2.0
                             (caddr(via_enc1) - cadr(cddr(via_enc1))) / 2.0 )
                off2 = list( (cadr(via_enc2) - car(via_enc2)) / 2.0
                             (caddr(via_enc2) - cadr(cddr(via_enc2))) / 2.0 )

                via_params = list( list("cutRows" via->num_rows)
                                   list("cutColumns" via->num_cols)
                                   list("cutSpacing" list(via->sp_cols via->sp_rows))
                                   list("layer1Enc" enc1)
                                   list("layer2Enc" enc2)
                                   list("layer1Offset" off1)
                                   list("layer2Offset" off2) )

                ; if via width and height given, add to via_params
                when( via->cut_width != nil
                    via_params = cons( list("cutWidth" via->cut_width) via_params)
                )
                when( via->cut_height != nil
                    via_params = cons( list("cutHeight" via->cut_height) via_params)
                )

                ; create actual via
                orig_shape = dbCreateVia(cv via_def via->loc via->orient via_params)
                if( not(orig_shape) then
                    warn("Error creating via %s.  Skipping." via->id)
                else
                    when( via->arr_nx != nil
                        for(icol 2 via->arr_nx
                            arr_dx = via->arr_spx * (icol - 1)
                            for(irow 1 via->arr_ny
                                arr_dy = via->arr_spy * (irow - 1)
                                dbCopyFig(orig_shape nil list(arr_dx:arr_dy "R0" 1))
                            )
                        )
                        for(irow 2 via->arr_ny
                            arr_dy = via->arr_spy * (irow - 1)
                            dbCopyFig(orig_shape nil list(0:arr_dy "R0" 1))
                        )
                    )
                )
            else
                warn("Via %s not found.  Skipping." via->id)
            )
        )

        ; create pins
        foreach( pin pin_list
            pin_bb = pin->bbox
            pin_w = caadr(pin_bb) - caar(pin_bb)
            pin_h = cadr(cadr(pin_bb)) - cadr(car(pin_bb))
            pin_xc = (caar(pin_bb) + caadr(pin_bb)) / 2.0
            pin_yc = (cadr(car(pin_bb)) + cadr(cadr(pin_bb))) / 2.0

            if( pin_w >= pin_h then
                pin_orient = "R0"
                label_h = pin_h
            else
                pin_orient = "R90"
                label_h = pin_w
            )

            ; get make_pin_rect, true if both net_name and pin_name are non-empty
            make_pin_rect = pin->net_name != "" && pin->pin_name != ""
            when( pin->make_rect != nil
                make_pin_rect = pin->make_rect
            )
            ; printf("make_pin_rect: %A\n" make_pin_rect)
            ; create pin object only if make_pin_rect is True.
            when( make_pin_rect != 0 && make_pin_rect != nil
                ; printf("making pin.\n")
                dbCreatePin( dbMakeNet(cv pin->net_name) dbCreateRect(cv pin->layer pin_bb) pin->pin_name )
            )
            ; printf("%A %A %A %A\n" pin->label pin->layer pin_xc pin_yc)
            dbCreateLabel( cv pin->layer list(pin_xc pin_yc) pin->label "centerCenter" pin_orient "roman" label_h )
        )
    )
)

; create a new layout view with the given geometries
; inst_f, rect_f, via_f, and pin_f are files containing list of disembodied property lists.
procedure( create_layout( lib_name view_name via_tech layout_f "ttt" )
    let( (tech_file layout_info cell_name inst_list rect_list via_list pin_list
          path_list blockage_list boundary_list polygon_list cv)

        unless( tech_file = techGetTechFile(ddGetObj(via_tech))
            error("Via technology file %s not found." via_tech)
        )

        layout_info = parse_data_from_file(layout_f)
        foreach( info layout_info
            cell_name = nthelem(1 info)
            inst_list = nthelem(2 info)
            rect_list = nthelem(3 info)
            via_list = nthelem(4 info)
            pin_list = nthelem(5 info)
            path_list = nthelem(6 info)
            blockage_list = nthelem(7 info)
            boundary_list = nthelem(8 info)
            polygon_list = nthelem(9 info)

            unless( cv = dbOpenCellViewByType( lib_name cell_name view_name "maskLayout" "w" )
                error("Cannot create new layout cell %s__%s (%s)." lib_name cell_name view_name)
            )

            printf("Creating %s__%s (%s)\n" lib_name cell_name view_name)
            create_layout_helper(cv tech_file inst_list rect_list via_list pin_list path_list
                                 blockage_list boundary_list polygon_list)

            dbSave(cv)
            dbClose(cv)
        )

        t
    )
)

; release write locks from all the given cellviews
procedure( release_write_locks( lib_name cell_view_list_f "tt" )
    let( (cell_view_list lib_obj cv)
        cell_view_list = parse_data_from_file(cell_view_list_f)
        when( lib_obj = ddGetObj(lib_name nil nil nil nil "r")
            foreach( info cell_view_list
                when( cv = dbFindOpenCellView( lib_obj car(info) cadr(info) )
                    dbReopen(cv, "r")
                    dbClose(cv)
                )
            )
            ddReleaseObj(lib_obj)
        )
        t
    )
)


;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;  Simulation/Testbench related functions  ;;
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;


; set an entry in an association list
; returns the modified association list.
procedure( set_assoc_list(mylist mykey myval)
    let( (tmp)
        when( tmp = assoc(mykey mylist)
            ; print("replacing")
            rplacd(tmp list(myval))
        )
    )
    mylist
)

; Copy the schematic of a testbench, and replace the DUT instance.
;
; This procedure copies the schematic of a testbench to a new library and cell, then finds all
; instances with the name prefix "XDUT", then change their instance master to dut_lib and dut_cell.
;
procedure( copy_testbench(master_lib master_cell targ_lib
                          dut_lib dut_cell tech_lib new_lib_path "ttttttt")
    let( (tlib_obj sch replace_count inst_prefix new_master)
        inst_prefix = "XDUT"

        printf("Copying testbench %s__%s to %s__%s\n" master_lib master_cell targ_lib master_cell)

        ; create target library if does not exist
        unless( tlib_obj = ddGetObj(targ_lib nil nil nil nil "r")
            when( and(new_lib_path (new_lib_path != "."))
                new_lib_path = strcat(new_lib_path "/" lib_name)
            )
            tlib_obj = ddCreateLib(targ_lib new_lib_path)
            ; attach technology file
            techBindTechFile(tlib_obj tech_lib)
        )

        ; copy testbench to new library
        src_gdm = gdmCreateSpecList()
        gdmAddSpecToSpecList(gdmCreateSpec(master_lib master_cell nil nil "CDBA") src_gdm)
        targ_gdm = gdmCreateSpecList()
        gdmAddSpecToSpecList(gdmCreateSpec(targ_lib master_cell nil nil "CDBA") targ_gdm)
        ccpCopy(src_gdm targ_gdm 't 'CCP_EXPAND_COMANAGED)

        ; open copied schematic
        unless( sch = dbOpenCellViewByType(tlib_obj master_cell "schematic" nil "a")
            ddReleaseObj(tlib_obj)
            error("Cannot open testbench schematic %s__%s" targ_lib master_cell)
        )

        ; replace instances
        replace_count = 0
        sprintf(new_master "%s %s symbol" dut_lib dut_cell)
        foreach( inst sch~>instances
           when( strncmp( inst~>name inst_prefix strlen(inst_prefix) ) == 0
               replace_count = replace_count + 1
               schReplaceProperty(list(inst) "master" new_master)
           )
        )

        ; save and close resources
        check_and_save_schematic(sch)
        ddReleaseObj(tlib_obj)

        ; error if nothing is replaced
        when( replace_count == 0
            error("Cannot find any instances in %s__%s with name prefix %s" targ_lib master_cell inst_prefix)
        )
        't
    )
)

; opens an adexl session.  Returns a list of session name and setup database handle.
procedure( open_adexl_session(tb_lib tb_cell tb_view session_name mode "ttttt")
    let( (session sdb)
        unless( session = axlCreateSession(session_name)
            error("Cannot create temporary adexl session: %s" session_name)
        )
        unless( sdb = axlSetMainSetupDBLCV(session tb_lib tb_cell tb_view ?mode mode)
            axlCloseSession(session)
            error("Cannot load adexl database from %s__%s (%s)" tb_lib tb_cell tb_view)
        )
        list(session sdb)
    )
)

; Enables only the given corners in the simulation setup database.
procedure( enable_adexl_corners( sdb corner_list env_param_list "gll")
    let( (env_name par_val_list corner)
        foreach(cur_name cadr(axlGetCorners(sdb))
            axlSetEnabled( axlGetCorner(sdb cur_name) member(cur_name corner_list) )
        )
        foreach(env_par_obj env_param_list
            env_name = car(env_par_obj)
            par_val_list = cadr(env_par_obj)
            corner = axlGetCorner(sdb env_name)
            foreach(par_val par_val_list
                axlPutVar(corner car(par_val) cadr(par_val))
            )
        )
    )
)

; Set testbench parameters
; val_list is an association list from variable names to variable values as string, which
; could be a constant value or a parametric sweep string
procedure( set_adexl_parameters(sdb par_val_list "gl")
    foreach( var_spec par_val_list
        axlPutVar(sdb car(var_spec) cadr(var_spec))
    )
)

; Create a new config view for a testbench.
;
; lib_name : testbench library name.
; cell_name : testbench cell name.
; view_name : name of the config view (a testbench can have multiple config views)
; libs : a string of global libraries, separated by spaces.
; views : a string of cellviews to use, separated by spaces.
; stops : a string of cellviews to stop at, separated by spaces.
procedure( create_config_view(lib_name cell_name view_name libs views stops "tttttt")
    let( (conf conf_bag)
        printf("Creating config view %s__%s (%s)\n" lib_name cell_name view_name)

        unless( conf = hdbOpen(lib_name cell_name view_name "w")
            error("Cannot open config view %s__%s (%s)." lib_name cell_name view_name)
        )
        hdbSetTopCellViewName(conf lib_name cell_name "schematic")
        hdbSetDefaultLibListString(conf libs)
        hdbSetDefaultViewListString(conf views)
        hdbSetDefaultStopListString(conf stops)
        hdbSaveAs(conf lib_name cell_name view_name)

        ; close configuration
        conf_bag = hdbCreateConfigBag()
        hdbAddConfigToBag(conf_bag conf)
        hdbCloseConfigsInBag(conf_bag)
    )
)

; edit the config view of a testbench.  Use to control whether we're simulating with
; schematic or post-extraction.
;
; lib_name : testbench library name.
; cell_name : testbench cell name.
; view_name : name of the config view (a testbench can have multiple config views)
; conf_list : a list of (<lib>, <cell>, <view>) configurations.  Where each entry
;             means that view <view> should be used for the cell <cell> in library <lib>.
procedure( edit_config_view(lib_name cell_name view_name conf_list "tttl")
    let( (conf lib cell view conf_bag netlist_list)
        unless( conf = hdbOpen(lib_name cell_name view_name "a")
            error("Cannot open config view %s__%s (%s)." lib_name cell_name view_name)
        )
        netlist_list = '()
        foreach( cell_config conf_list
            lib = car(cell_config)
            cell = cadr(cell_config)
            view = caddr(cell_config)
            if( view == "netlist" then
                ; set to use extracted netlist
                netlist_list = cons(list(lib cell) netlist_list)
            else
                ; set to use extracted cellview
                hdbSetObjBindRule(conf list(list(lib cell nil nil))
                                  list('hdbcBindingRule list(nil nil view)))
            )
        )
        hdbSaveAs(conf lib_name cell_name view_name)

        ; close configuration
        conf_bag = hdbCreateConfigBag()
        hdbAddConfigToBag(conf_bag conf)
        hdbCloseConfigsInBag(conf_bag)

        ; update netlist source files
        edit_config_source_files(lib_name cell_name view_name netlist_list)
    )
)

; HACKERMAN FUNCTION:
; so as usual, cadence is so terrible they don't have skill API to set source files.
; instead, spice/spectre source files are defined in a secret ASCII prop.cfg file.
; this hacky method will create the right prop.cfg file for you.
procedure( edit_config_source_files(lib_name cell_name view_name netlist_list "tttl")
    let( (p lib_dir cell_lib_dir)
        lib_dir = get_lib_directory(lib_name)
        p = outfile( sprintf(nil "%s/%s/%s/%s" lib_dir cell_name view_name "prop.cfg") "w" )
        ; common header
        fprintf( p "file-format-id 1.1;\ndefault\n{\n}\n" )
        foreach( lib_cell netlist_list
            lib = car(lib_cell)
            cell = cadr(lib_cell)
            cell_lib_dir = get_lib_directory(lib)
            fprintf( p "cell %s.%s\n{\n" lib cell )
            fprintf( p "    non-inherited string prop sourcefile = \"%s/%s/netlist/netlist\";\n}\n"
                     cell_lib_dir cell )
        )
        close(p)
    )
)

; Write testbench information to file.
procedure( write_testbench_info_to_file(sdb result_file output_list en_corner_list)
    let( (p output_count)

        ; write testbench information to result_file
        p = outfile(result_file "w")

        fprintf(p "corners:\n")
        foreach( corn cadr(axlGetCorners(sdb))
            fprintf(p "  - %s\n" corn)
        )
        fprintf(p "enabled_corners:\n")
        foreach( corn en_corner_list
            fprintf(p "  - %s\n" corn)
        )
        fprintf(p "parameters:\n")
        if( var_list = cadr(axlGetVars(sdb)) then
            foreach( var_name var_list
                fprintf(p "  %s: \"%s\"\n" var_name axlGetVarValue(axlGetVar(sdb var_name)))
            )
        else
            fprintf(p "  {}\n")
        )
        fprintf(p "outputs:\n")
        output_count = 0
        foreach( out_obj output_list
            if( rexMatchp( "\"" out_obj->name) then
                warn("Output expression name (%s) have quotes, skipping" out_obj->name)
            else
                fprintf(p "  \"%s\": !!str %A\n" out_obj->name out_obj->expression)
                output_count = output_count + 1
            )
        )
        when( output_count == 0
            fprintf(p "  {}\n")
        )
        close(p)
    )
)

; Instantiates a testbench.
;
; Copy a testbench template to the desired location, replace instances, make config view,
; and also setup corner settings in adexl.
; this method will also record list of corners, global variables, and output expressions
; to result_file
procedure( instantiate_testbench(tb_cell targ_lib
                                 config_libs config_views config_stops
                                 default_corner corner_file def_files
                                 tech_lib result_file
                                 "tttttttltt")
    let( (session_name session_sdb session sdb test_names test_name test tool_args corner_list
          ade_symbol ade_session output_list tmp_state_name state_obj success)

        tmp_state_name = "orig_state"

        ; check if temporary ADE session state already exists, if so, delete it
        state_obj = ddGetObj(targ_lib tb_cell tmp_state_name)
        when( state_obj
            success = ddDeleteObj(state_obj)
            unless( success
                error("Cannot delete orig_state cellview.")
            )
        )

        ; create config view
        create_config_view(targ_lib tb_cell "config" config_libs config_views config_stops)

        ; session_name = "modify_adexl"
        session_name = sprintf(nil "modify_adexl_%d" bag_modify_adexl_counter)
        bag_modify_adexl_counter = bag_modify_adexl_counter + 1

        session_sdb = open_adexl_session(targ_lib tb_cell "adexl" session_name "a")
        session = car(session_sdb)
        sdb = cadr(session_sdb)

        ; check that only one test is defined
        test_names = cadr(axlGetTests(sdb))
        when(length(test_names) != 1
            axlCommitSetupDB(sdb)
            axlCloseSetupDB(sdb)
            axlCloseSession(session)
            error("ADEXL testbench must have exactly 1 test defined.")
        )

        ; save current test setup state
        axlSaveSetupState(session "adexl_default" "All")

        ; change all tests to use config view, and set all test's definition files
        ; also get a list of defined output expressions
        ; step 1: get ADE session
        test_name = car(test_names)
        ade_symbol = axlGetToolSession(session_name test_name)
        ade_session = asiGetSession(ade_symbol)
        ; step 2: save original ADE session
        asiSaveState(ade_session ?name tmp_state_name ?option 'cellview ?lib targ_lib ?cell tb_cell)
        ; step 3: change test library
        test = axlGetTest(sdb test_name)
        tool_args = axlGetTestToolArgs(test)
        set_assoc_list(tool_args "view" "config")
        set_assoc_list(tool_args "lib" targ_lib)
        set_assoc_list(tool_args "cell" tb_cell)
        axlSetTestToolArgs(test tool_args)
        ; step 4: reopen ADE session, then load original ADE state
        ade_symbol = axlGetToolSession(session_name test_name)
        ade_session = asiGetSession(ade_symbol)
        asiLoadState(ade_session ?name tmp_state_name ?option 'cellview)
        asiSetEnvOptionVal(ade_session 'definitionFiles def_files)
        output_list = setof(ele asiGetOutputList(ade_session) ele->name)
        ; step 5: delete temporary ADE session state
        state_obj = ddGetObj(targ_lib tb_cell tmp_state_name)
        ddDeleteObj(state_obj)

        axlMainAppSaveSetup(session_name)

        ; load corner
        unless(axlLoadCorners(sdb corner_file)
            axlCommitSetupDB(sdb)
            axlCloseSetupDB(sdb)
            axlCloseSession(session)
            error("Error loading corner file %s to %s__%s (%s)" corner_file lib_name cell_name view_name)
        )

        ; set default corner
        corner_list = list(default_corner)
        enable_adexl_corners(sdb corner_list nil)

        ; write testbench information to file
        write_testbench_info_to_file(sdb result_file output_list corner_list)

        ; save and close
        axlSaveSetupState(session "adexl_default" "All")
        axlSaveSetupState(session "ocean_default" "All")
        axlMainAppSaveSetup(session_name)
        axlCommitSetupDB(sdb)
        axlCloseSetupDB(sdb)
        axlCloseSession(session)
    )
)

; Returns parameter and corner information of a testbench.
procedure( get_testbench_info(tb_lib tb_cell result_file "ttt")
    let( (session_name session_sdb session sdb test_names test_name ade_symbol asi_sess
        output_list corner_list en_list success)
        session_name = "read_adexl"
        session_sdb = open_adexl_session(tb_lib tb_cell "adexl" session_name "r")
        session = car(session_sdb)
        sdb = cadr(session_sdb)

        ; check that only one test is defined
        test_names = cadr(axlGetTests(sdb))
        when(length(test_names) != 1
            axlCommitSetupDB(sdb)
            axlCloseSetupDB(sdb)
            axlCloseSession(session)
            error("ADEXL testbench must have exactly 1 test defined.")
        )

        ; get output list
        test_name = car(test_names)
        ade_symbol = axlGetToolSession(session_name test_name)
        asi_sess = sevEnvironment(ade_symbol)
        output_list = setof(ele asiGetOutputList(asi_sess) ele->name)

        ; get enabled corners
        corner_list = cadr(axlGetCorners(sdb))
        en_list = setof(corner corner_list axlGetEnabled(axlGetCorner(sdb corner)))

        ; write testbench information to file
        write_testbench_info_to_file(sdb result_file output_list en_list)

        ; close
        axlCommitSetupDB(sdb)
        axlCloseSetupDB(sdb)
        axlCloseSession(session)
    )
)

; Configure run options.  Used to setup monte carlo parameters.
; run_params is an association list of run options and their values.  The key "mode"
; corresponds to the run mode.
procedure( set_run_options(session sdb run_params "ggl")
    let( (run_mode opt_list run_opt)
        when( run_mode = cadr(assoc("mode" run_params))
                  ; no options for single run/sweep mode.
            cond( (run_mode == "Single Run, Sweeps and Corners"
                   opt_list = nil)
                  (run_mode == "Monte Carlo Sampling"
                   opt_list = '("mcnumpoints" "mcmethod") )
                  ('t
                      axlCloseSession(session)
                      error("Unsupported run mode: %s" run_mode) )
            )
            foreach( opt_name opt_list
                when( opt_val = cadr(assoc(opt_name run_params))
                    run_opt = axlPutRunOption(sdb run_mode opt_name)
                    axlSetRunOptionValue(run_opt opt_val)
                )
            )
            axlSetCurrentRunMode(sdb run_mode)
        )
    )
)

; modify the given testbench.
; tb_lib and tb_cell describes the library and cell of the testbench to simulate.
; conf_file contains the config view settings.
; opt_file contains the association list of run mode options.
; corner_file contains a list of corners to simulate.
; param_file contains the association list of parameter values.
procedure( modify_testbench(tb_lib tb_cell conf_file opt_file corner_file param_file env_params_file "ttttttt")
    let( (tmp_list session sdb conf_list run_params corner_list param_values env_param_values session_name)
        sprintf(session_name "bag_sim_adexl_%s" getCurrentTime())

        ; read inputs from file.
        conf_list = parse_data_from_file(conf_file)
        run_params = parse_data_from_file(opt_file)
        corner_list = parse_data_from_file(corner_file)
        param_values = parse_data_from_file(param_file)
        env_param_values = parse_data_from_file(env_params_file)

        ; modify config view
        when( conf_list
            edit_config_view(tb_lib tb_cell "config" conf_list)
        )

        tmp_list = open_adexl_session(tb_lib tb_cell "adexl" session_name "a")
        session = car(tmp_list)
        sdb = cadr(tmp_list)

        ; change corners, parameters, and run options
        enable_adexl_corners( sdb corner_list env_param_values)
        set_adexl_parameters( sdb param_values )
        set_run_options( session sdb run_params )

        ; save and close
        axlSaveSetupState(session "adexl_default" "All")
        axlSaveSetupState(session "ocean_default" "All")
        axlMainAppSaveSetup(session_name)
        axlCommitSetupDB(sdb)
        axlCloseSetupDB(sdb)
        axlCloseSession(session)
    )
)

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;  BAG server related functions            ;;
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

procedure( stdoutHandler(ipcId data)
    let( (result result_str)
        if( bag_server_started > 0 then
            printf("*INFO* Evaluate expression from BAG process: %s\n" data)
            if( result = errsetstring(data 't) then
                sprintf(result_str "%A\n" car(result))
            else
                sprintf(result_str "%s\n" car(nthelem(5 errset.errset)))
            )
            printf("*INFO* Sending result to BAG process: %s" result_str)
            ipcWriteProcess(ipcId sprintf(nil "%d\n" strlen(result_str)))
            ipcWriteProcess(ipcId result_str)
            't
        else
            if( data == "BAG skill server has started.  Yay!\n" then
                bag_server_started = 1
                printf("*INFO* BAG skill server started.\n")
            else
                printf("*INFO* Waiting for BAG skill server.  Message: %s\n" data)
            )
        )
    )
)

procedure( stderrHandler(ipcId data)
    warn("BAG server process error: %s\n" data)
    warn("Shutting down BAG server.")
    ipcKillProcess(ipcId)
    't
)

procedure( exitHandler(ipcId exitId)
    printf("*INFO* BAG server process exited with status: %d\n" exitId)
    't
)

procedure( start_bag_server()
    bag_server_started = 0
    printf("*INFO* Starting BAG server process.\n")
    ipcBeginProcess("bash virt_server.sh" "" 'stdoutHandler 'stderrHandler 'exitHandler "")
)

bag_server_started = 0
bag_modify_adexl_counter = 0
bag_proc = start_bag_server()
