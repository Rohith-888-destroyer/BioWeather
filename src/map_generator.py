import json
import os
import shutil
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

def generate_interactive_map(df, output_html="docs/latest_map.html"):
    """Generate interactive Plotly scatter-geo map of outbreak risk tiers."""
    os.makedirs(os.path.dirname(output_html), exist_ok=True)
    
    color_map = {
        "High": "#ff4757",    # Bright Red
        "Medium": "#ffa502",  # Orange
        "Low": "#2ed573"      # Green
    }

    fig = px.scatter_geo(
        df,
        lat="lat",
        lon="lon",
        color="risk_tier",
        hover_name="name",
        hover_data={
            "country": True,
            "endemic_focus": True,
            "mean_temp_14d": ":.1f °C",
            "mean_humidity_14d": ":.1f %",
            "risk_score": ":.2f",
            "lat": False,
            "lon": False
        },
        size="risk_score",
        size_max=18,
        color_discrete_map=color_map,
        projection="natural earth",
        title="BioWeather Global Vector-Borne Outbreak Risk Forecaster"
    )

    fig.update_layout(
        template="plotly_dark",
        margin=dict(l=0, r=0, t=50, b=0),
        geo=dict(
            showland=True,
            landcolor="#121824",
            showocean=True,
            oceancolor="#0a0e17",
            showlakes=True,
            lakecolor="#0a0e17",
            showcountries=True,
            countrycolor="#1e293b"
        ),
        legend=dict(
            title="Outbreak Risk Tier",
            yanchor="top",
            y=0.95,
            xanchor="left",
            x=0.01,
            bgcolor="rgba(10, 14, 23, 0.8)"
        )
    )

    fig.write_html(output_html)
    # Also save to root for zero-config Vercel deployment
    fig.write_html("latest_map.html")
    print(f"Saved interactive map: {output_html} & latest_map.html")
    return fig

def generate_static_png(fig, df, output_png="docs/latest_map.png"):
    """Render static PNG snapshot for README embedding."""
    os.makedirs(os.path.dirname(output_png), exist_ok=True)
    try:
        fig.write_image(output_png, width=1200, height=630, scale=2)
        fig.write_image("latest_map.png", width=1200, height=630, scale=2)
        print(f"Successfully generated static PNG via Plotly/kaleido: {output_png}")
    except Exception as e:
        print(f"Plotly write_image unavailable ({e}). Generating fallback static image...")
        from PIL import Image, ImageDraw

        img = Image.new("RGB", (1200, 630), color="#0a0e17")
        draw = ImageDraw.Draw(img)

        # Title banner
        draw.rectangle([(0, 0), (1200, 70)], fill="#121824")
        draw.text((30, 25), "BioWeather Global Outbreak Risk Forecast Map", fill="#ffffff")

        color_map = {"High": (255, 71, 87), "Medium": (255, 165, 2), "Low": (46, 213, 115)}
        for _, row in df.iterrows():
            x = int(50 + ((row["lon"] + 180) / 360.0) * 1100)
            y = int(550 - ((row["lat"] + 90) / 180.0) * 450)
            r = int(6 + row["risk_score"] * 12)
            c = color_map.get(row["risk_tier"], (255, 255, 255))
            draw.ellipse([x - r, y - r, x + r, y + r], fill=c, outline="#ffffff")

        img.save(output_png)
        img.save("latest_map.png")
        print(f"Saved fallback static PNG: {output_png} & latest_map.png")

def export_embedded_js_data(df, output_js="docs/data_embedded.js"):
    """Export dataset as JS variable for standalone web app fallback."""
    records = df.to_dict(orient="records")
    js_content = f"window.EMBEDDED_BIO_DATA = {json.dumps(records, indent=2)};"
    with open(output_js, "w") as f:
        f.write(js_content)
    with open("data_embedded.js", "w") as f:
        f.write(js_content)
    print(f"Saved embedded web app data payload: {output_js} & data_embedded.js")

def run_map_generation(csv_path="data/processed/risk_predictions.csv", docs_dir="docs"):
    """Load risk predictions and generate interactive and static map artifacts."""
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Predictions CSV {csv_path} not found. Run infer.py first.")

    df = pd.read_csv(csv_path)
    output_html = os.path.join(docs_dir, "latest_map.html")
    output_png = os.path.join(docs_dir, "latest_map.png")

    fig = generate_interactive_map(df, output_html=output_html)
    generate_static_png(fig, df, output_png=output_png)
    export_embedded_js_data(df, os.path.join(docs_dir, "data_embedded.js"))

    # Copy CSS and JS to root for zero-config Vercel deployment
    shutil.copy(os.path.join(docs_dir, "styles.css"), "styles.css")
    shutil.copy(os.path.join(docs_dir, "app.js"), "app.js")
    print("Map and Web App artifact generation completed successfully.")

if __name__ == "__main__":
    run_map_generation()
