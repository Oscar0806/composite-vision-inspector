import gradio as gr
import cv2
import numpy as np
from detector import inspect_surface
 
def run_inspection(input_image, threshold, min_area):
    """Gradio wrapper for the inspection function."""
    if input_image is None:
        return None, "No image uploaded", ""
    
    # Gradio gives RGB numpy array, OpenCV needs BGR
    img_bgr = cv2.cvtColor(input_image, cv2.COLOR_RGB2BGR)
    
    # Run inspection
    result = inspect_surface(img_bgr, threshold=int(threshold),
                             min_defect_area=int(min_area))
    
    # Convert annotated image back to RGB for Gradio
    annotated_rgb = cv2.cvtColor(result["annotated_image"],
                                  cv2.COLOR_BGR2RGB)
    
    # Build status text
    if result["status"] == "PASS":
        status = "\u2705 PASS \u2014 No defects detected"
    else:
        status = (f"\u274C FAIL \u2014 {result['defect_count']} "
                  f"defect(s) found")
    
    # Build detailed report
    report = f"Status: {result['status']}\n"
    report += f"Defects found: {result['defect_count']}\n"
    report += (f"Total defect area: "
               f"{result['total_defect_area_px']} pixels\n")
    report += "\n"
    
    if result["defects"]:
        report += "Defect details:\n"
        for i, d in enumerate(result["defects"], 1):
            report += (f"  #{i}: {d['type']} | "
                       f"Area: {d['area_px']}px | "
                       f"Location: ({d['location'][0]}, "
                       f"{d['location'][1]}) | "
                       f"Circularity: {d['circularity']}\n")
    else:
        report += "Surface is clean. No defects detected."
    
    return annotated_rgb, status, report
 
# ── BUILD INTERFACE ──
with gr.Blocks(
    title="Composite Surface Inspector",
    theme=gr.themes.Soft()
) as app:
    
    gr.Markdown(
        "# \u2708\uFE0F Composite Surface Quality Inspector\n"
        "Upload a CFRP composite surface image to detect "
        "defects (scratches, dents, delamination).\n\n"
        "*Concept for Airbus Stade automated quality "
        "inspection \u2014 Built by Oscar Vincent Dbritto*"
    )
    
    with gr.Row():
        with gr.Column():
            input_img = gr.Image(label="Upload surface image",
                                  type="numpy")
            with gr.Row():
                thresh_slider = gr.Slider(
                    30, 120, value=65, step=5,
                    label="Detection threshold",
                    info="Lower = more sensitive")
                area_slider = gr.Slider(
                    10, 200, value=50, step=10,
                    label="Min defect area (px)",
                    info="Filter out small noise")
            inspect_btn = gr.Button("\U0001f50d Inspect Surface",
                                     variant="primary")
        
        with gr.Column():
            output_img = gr.Image(label="Inspection result")
            status_text = gr.Textbox(label="Verdict",
                                      interactive=False)
            report_text = gr.Textbox(label="Detailed report",
                                      lines=8,
                                      interactive=False)
    
    inspect_btn.click(
        fn=run_inspection,
        inputs=[input_img, thresh_slider, area_slider],
        outputs=[output_img, status_text, report_text]
    )
    
    # Example images
    gr.Markdown("### Example images")
    gr.Examples(
        examples=[
            ["test_images/good/good_001.png", 65, 50],
            ["test_images/good/good_005.png", 65, 50],
            ["test_images/defect/defect_001.png", 65, 50],
            ["test_images/defect/defect_003.png", 65, 50],
            ["test_images/defect/defect_010.png", 65, 50],
        ],
        inputs=[input_img, thresh_slider, area_slider],
    )
    
    gr.Markdown(
        "---\n"
        "**Defect types:** "
        "\U0001f534 Scratch (red) | "
        "\U0001f7e0 Dent/Porosity (orange) | "
        "\U0001f7e1 Delamination (yellow)\n\n"
        "Built by Oscar Vincent Dbritto | "
        "M.Sc. Digitalization & Automation | "
        "Concept for Airbus CFRP quality inspection"
    )
 
if __name__ == "__main__":
    app.launch()
