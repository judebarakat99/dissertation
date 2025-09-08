#!/usr/bin/env python3
import re
import rclpy
from rclpy.node import Node

def sid(s: str) -> str:
    return re.sub(r'[^A-Za-z0-9_]', '_', s)

rclpy.init()
node = Node('ros2graph_to_mermaid')
labels = {}
edges = []

for name, ns in node.get_node_names_and_namespaces():
    full = (ns.rstrip('/') + '/' + name) if ns and ns != '/' else '/' + name
    nid = sid(full); labels[nid] = full
    for topic, _ in node.get_publisher_names_and_types_by_node(name, ns):
        tid = sid(topic); labels[tid] = topic; edges.append(f'{nid} --> {tid}')
    for topic, _ in node.get_subscriber_names_and_types_by_node(name, ns):
        tid = sid(topic); labels[tid] = topic; edges.append(f'{tid} --> {nid}')

print('%%{init: {"theme":"neutral","flowchart":{"curve":"linear"}}}%%')
print('flowchart LR')
for i,lbl in labels.items():
    print(f'{i}(["{lbl}"])' if lbl.startswith('/') else f'{i}["{lbl}"]')
for e in edges: print(e)

node.destroy_node()
rclpy.shutdown()
