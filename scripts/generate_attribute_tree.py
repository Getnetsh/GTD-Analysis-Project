from graphviz import Digraph

# Initialize directed graph with tree layout
dot = Digraph(comment='GTD Attribute Tree', format='png', graph_attr={'rankdir': 'TB', 'splines': 'ortho'})

# Fact (root)
dot.node('Fact', 'Terrorism Incidents', shape='box', style='filled', fillcolor='lightblue')

# Measures branch
dot.node('Measures', 'Measures', shape='box', style='filled', fillcolor='lightyellow')
dot.node('incident_count', 'incident_count', shape='ellipse')
dot.node('nkill', 'nkill', shape='ellipse')
dot.node('nwound', 'nwound', shape='ellipse')
dot.node('nkillter', 'nkillter', shape='ellipse')
dot.node('nwoundter', 'nwoundter', shape='ellipse')
dot.node('propvalue', 'propvalue', shape='ellipse')
dot.node('success', 'success', shape='ellipse')

# Dimensions branch
dot.node('Dimensions', 'Dimensions', shape='box', style='filled', fillcolor='lightyellow')

# Time dimension
dot.node('Time', 'Time', shape='box', style='filled', fillcolor='lightgreen')
dot.node('year', 'year', shape='ellipse')
dot.node('month', 'month', shape='ellipse')
dot.node('day', 'day', shape='ellipse')
dot.node('approxdate', 'approxdate', shape='ellipse')
dot.node('extended', 'extended', shape='ellipse')

# Location dimension
dot.node('Location', 'Location', shape='box', style='filled', fillcolor='lightgreen')
dot.node('region', 'region', shape='ellipse')
dot.node('country', 'country', shape='ellipse')
dot.node('provstate', 'provstate', shape='ellipse')
dot.node('city', 'city', shape='ellipse')
dot.node('latitude', 'latitude', shape='ellipse')
dot.node('longitude', 'longitude', shape='ellipse')
dot.node('specificity', 'specificity', shape='ellipse')
dot.node('vicinity', 'vicinity', shape='ellipse')

# Attack Type dimension
dot.node('AttackType', 'Attack Type', shape='box', style='filled', fillcolor='lightgreen')
dot.node('attack_type', 'attack_type', shape='ellipse')
dot.node('weapon_type', 'weapon_type', shape='ellipse')
dot.node('weapon_subtype', 'weapon_subtype', shape='ellipse')

# Target Type dimension
dot.node('TargetType', 'Target Type', shape='box', style='filled', fillcolor='lightgreen')
dot.node('target_type', 'target_type', shape='ellipse')
dot.node('target_subtype', 'target_subtype', shape='ellipse')
dot.node('nationality', 'nationality', shape='ellipse')

# Terrorist Group dimension
dot.node('TerroristGroup', 'Terrorist Group', shape='box', style='filled', fillcolor='lightgreen')
dot.node('group_name', 'group_name', shape='ellipse')
dot.node('subgroup_name', 'subgroup_name', shape='ellipse')
dot.node('uncertainty', 'uncertainty', shape='ellipse')

# Property Damage dimension
dot.node('PropertyDamage', 'Property Damage', shape='box', style='filled', fillcolor='lightgreen')
dot.node('extent', 'extent', shape='ellipse')
dot.node('property', 'property', shape='ellipse')

# Edges for tree structure
dot.edge('Fact', 'Measures')
dot.edge('Fact', 'Dimensions')

# Measures edges
dot.edge('Measures', 'incident_count')
dot.edge('Measures', 'nkill')
dot.edge('Measures', 'nwound')
dot.edge('Measures', 'nkillter')
dot.edge('Measures', 'nwoundter')
dot.edge('Measures', 'propvalue')
dot.edge('Measures', 'success')

# Dimensions edges
dot.edge('Dimensions', 'Time')
dot.edge('Dimensions', 'Location')
dot.edge('Dimensions', 'AttackType')
dot.edge('Dimensions', 'TargetType')
dot.edge('Dimensions', 'TerroristGroup')
dot.edge('Dimensions', 'PropertyDamage')

# Time hierarchy
dot.edge('Time', 'year')
dot.edge('year', 'month')
dot.edge('month', 'day')
dot.edge('Time', 'approxdate')
dot.edge('Time', 'extended')

# Location hierarchy
dot.edge('Location', 'region')
dot.edge('region', 'country')
dot.edge('country', 'provstate')
dot.edge('provstate', 'city')
dot.edge('Location', 'latitude')
dot.edge('Location', 'longitude')
dot.edge('Location', 'specificity')
dot.edge('Location', 'vicinity')

# Attack Type hierarchy
dot.edge('AttackType', 'attack_type')
dot.edge('attack_type', 'weapon_type')
dot.edge('weapon_type', 'weapon_subtype')

# Target Type hierarchy
dot.edge('TargetType', 'target_type')
dot.edge('target_type', 'target_subtype')
dot.edge('target_subtype', 'nationality')

# Terrorist Group (flat)
dot.edge('TerroristGroup', 'group_name')
dot.edge('TerroristGroup', 'subgroup_name')
dot.edge('TerroristGroup', 'uncertainty')

# Property Damage (flat)
dot.edge('PropertyDamage', 'extent')
dot.edge('PropertyDamage', 'property')

# Save the diagram
dot.render('schemas/attribute_tree', cleanup=True)