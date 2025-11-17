#!/usr/bin/env python
# Modified from : https://gist.github.com/awesomebytes/a3bc8729d0c1d0a9499172b9a77d2622

import sys
import os
import subprocess

# Script taken from doing the needed operation
# And going to Filter > Show current filter script

filter_script_mlx = """<!DOCTYPE FilterScript>
<FilterScript>
 <filter name="Convex Hull">
  <Param type="RichBool" value="true" name="reorient"/>
 </filter>
</FilterScript>
"""


def create_tmp_filter_file(filename='filter_file_tmp.mlx'):
    with open('/tmp/' + filename, 'w') as f:
        f.write(filter_script_mlx)
    return '/tmp/' + filename


def apply_meshlab_filter(in_file, out_file,
                         filter_script_path=create_tmp_filter_file()):
    # Add input mesh
    command = "meshlabserver -i " + in_file
    # Add the filter script
    command += " -s " + filter_script_path
    # Add the output filename and output flags
    command += " -o " + out_file + " -om vn fn"
    # Execute command
    print( "Going to execute: " + command)
    output = subprocess.check_output(command, shell=True)
    last_line = output.splitlines()[-1]
    print( in_file + " > " + out_file + ": " + last_line)


if __name__ == '__main__':

    if len(sys.argv) < 2 :
        print("Applies the meshlab filter specified in the filterscript variable to the input files "
              "and saves the output with _convex added to the filename")
        print("Usage: ./generate_convexhulls.py [remove_old] <input files>")
        print("\tAdding \"remove_old\" as second argument replaces old meshes")
        print("Example to find all stl meshes in a directory and sub dirs:")
        print("\t$find . | grep .STL | grep -v \"_convex\" | xargs  ./generate_convex_hulls.py remove_old")

    start_index = 1
    if "remove_old" in sys.argv :
        start_index = 2

    for mesh in sys.argv[start_index:]:

        in_mesh = mesh
        filename = in_mesh.split('/')[-1]
        print("Input mesh: " + in_mesh + " (filename: " + filename + ")")

        out_mesh = in_mesh[:-4] + "_convex.STL"

        if os.path.isfile(out_mesh) and sys.argv[2] == "remove_old":
            os.remove(out_mesh)

        apply_meshlab_filter(in_mesh, out_mesh)

        print("Done with: "  + out_mesh)