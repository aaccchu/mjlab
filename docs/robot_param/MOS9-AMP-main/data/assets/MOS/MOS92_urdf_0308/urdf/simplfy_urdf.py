import os
import xml.etree.ElementTree as ET

import trimesh

# 你的 URDF 文件路径
urdf_path = "assets/MOS92_urdf_0308/urdf/MOS92_urdf_0308_base_link.urdf"
mesh_dir = "assets/MOS92_urdf_0308/meshes"  # 存放 stl 的目录

tree = ET.parse(urdf_path)
root = tree.getroot()

for link in root.findall("link"):
  collision = link.find("collision")
  if collision is not None:
    geometry = collision.find("geometry")
    mesh = geometry.find("mesh")

    if mesh is not None:
      # 1. 获取原 mesh 文件路径
      filename = mesh.get("filename")
      # 这里需要根据你的 filename 格式解析出真实路径
      # 假设 filename 是 "package://meshes/link1.stl"
      real_path = os.path.join(mesh_dir, filename.split("/")[-1])

      try:
        # 2. 使用 trimesh 加载模型计算尺寸
        mesh_obj = trimesh.load(real_path)
        extents = mesh_obj.bounding_box.extents  # [x_length, y_length, z_length]

        # 3. 假设连杆主要沿 Z 轴延伸 (你可以根据 extents 的最大值动态判断)
        length = extents[2]
        # 半径取 X 和 Y 范围的最大值的一半
        radius = max(extents[0], extents[1]) / 2.0

        # 4. 修改 URDF 树
        geometry.remove(mesh)
        cylinder = ET.SubElement(geometry, "cylinder")
        cylinder.set("radius", str(round(radius, 4)))
        cylinder.set("length", str(round(length, 4)))

        print(
          f"Link '{link.get('name')}' 碰撞体已替换为 Cylinder(r={radius:.3f}, l={length:.3f})"
        )

      except Exception as e:
        print(f"无法处理网格 {real_path}: {e}")

# 保存新的 URDF
tree.write(
  "assets/MOS92_urdf_0308/urdf/MOS92_urdf_0308_simplified.urdf",
  encoding="utf-8",
  xml_declaration=True,
)
