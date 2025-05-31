import plotly.express as px
import pandas as pd

# Prepare hierarchical data for the sunburst chart
data = {
    "labels": [],
    "parents": [],
    "colors": []
}

# Add the fact table "attack" as the root
data["labels"].append("attack")
data["parents"].append("")
data["colors"].append("#000000")  # Black for fact

# Define dimensions and their attributes based on Terrorist_db schema
dimensions = {
    "Time": ["year", "month", "day"],
    "Location": ["region", "country", "provstate", "city", "latitude", "longitude", 
                 "specificity", "vicinity", "unknown_location"],
    "Attack": ["attacktype1", "weaptype1", "weapsubtype1"],
    "Target": ["targtype1", "targsubtype1", "nationality1"],
    "Group": ["group_name", "uncertainty1", "unknown_group"],
    "Property": ["extent", "property", "propvalue_category"],
    "Incident": ["success", "extended", "propvalue_imputed", "us_casualty_imputed", "high_impact"]
}

# Add measures under a "Measures" pseudo-dimension
measures = {
    "Measures": ["nkill", "nwound", "nkillus", "nwoundus", "nkillter", "nwoundter", 
                 "propvalue", "propvalue_count", "attack_count"]
}

# Add dimensions and attributes to the data
for dim, attrs in dimensions.items():
    # Add the dimension
    data["labels"].append(dim)
    data["parents"].append("attack")
    data["colors"].append("#0000FF")  # Blue for dimensions
    # Add the attributes
    for attr in attrs:
        data["labels"].append(attr)
        data["parents"].append(dim)
        # Color location-related nodes green, others teal
        if attr in ["region", "country", "provstate", "city"]:
            data["colors"].append("#98FB98")  # Green for location
        else:
            data["colors"].append("#00CED1")  # Teal for attributes

# Add measures
for dim, attrs in measures.items():
    data["labels"].append(dim)
    data["parents"].append("attack")
    data["colors"].append("#0000FF")  # Blue for measures pseudo-dimension
    for attr in attrs:
        data["labels"].append(attr)
        data["parents"].append(dim)
        data["colors"].append("#00CED1")  # Teal for measures

# Add hierarchical relationships (e.g., year -> month -> day)
# For the sunburst, we need to adjust the parent relationships
data["labels"].extend(["month", "day"])
data["parents"].extend(["year", "month"])
data["colors"].extend(["#00CED1", "#00CED1"])  # Teal for attributes

data["labels"].extend(["country", "provstate", "city"])
data["parents"].extend(["region", "country", "provstate"])
data["colors"].extend(["#98FB98", "#98FB98", "#98FB98"])  # Green for location

data["labels"].extend(["weaptype1", "weapsubtype1"])
data["parents"].extend(["attacktype1", "weaptype1"])
data["colors"].extend(["#00CED1", "#00CED1"])  # Teal for attributes

data["labels"].extend(["targsubtype1", "nationality1"])
data["parents"].extend(["targtype1", "targsubtype1"])
data["colors"].extend(["#00CED1", "#00CED1"])  # Teal for attributes

# Create a DataFrame for Plotly
df = pd.DataFrame({
    "labels": data["labels"],
    "parents": data["parents"],
    "colors": data["colors"]
})

# Create the sunburst chart
fig = px.sunburst(
    df,
    names="labels",
    parents="parents",
    color="labels",
    color_discrete_map={label: color for label, color in zip(df["labels"], df["colors"])},
    title="Sunburst Attribute Tree for Terrorist_db (Centered on 'attack')"
)

# Update layout for better aesthetics
fig.update_layout(
    title_font_size=20,
    title_font_color="#333333",
    title_x=0.5,
    margin=dict(t=50, b=50, l=50, r=50),
    paper_bgcolor="white",
    plot_bgcolor="rgba(240, 248, 255, 0.8)",  # Light blue background
    sunburstcolorway=["#000000", "#0000FF", "#00CED1", "#98FB98"],  # Ensure color consistency
    extendsunburstcolors=False
)

# Update traces for better styling
fig.update_traces(
    textinfo="label",
    hoverinfo="label+percent parent",
    insidetextorientation="radial",  # Orient text radially for readability
    marker=dict(line=dict(color="DarkSlateGrey", width=1))
)

# Show the plot
fig.show()