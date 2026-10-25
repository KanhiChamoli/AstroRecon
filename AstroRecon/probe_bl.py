import bpy
m = bpy.data.materials.new("probe")
m.use_nodes = True
p = [n for n in m.node_tree.nodes if n.type == "BSDF_PRINCIPLED"][0]
print("BLENDER", bpy.app.version_string)
print("SOCKETS:", [s.name for s in p.inputs])
print("HAS_EMISSION:", "Emission" in [s.name for s in p.inputs])
print("HAS_EMISSION_COLOR:", "Emission Color" in [s.name for s in p.inputs])
