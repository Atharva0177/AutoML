#!/usr/bin/env python3
"""
AutoML Social Card PNG Generator

This script converts the social-card.svg to PNG format in multiple sizes.
It creates both standard (1280x640) and high-resolution (@2x: 2560x1280) versions.

Requirements:
    Windows: Use generate-social-card.html (recommended) or install GTK+
    Linux/Mac: pip install cairosvg pillow

Usage:
    python convert-to-png.py
"""

import os
from pathlib import Path

def convert_svg_to_png():
    """Convert SVG social card to PNG format."""
    
    # Get the script directory
    script_dir = Path(__file__).parent
    svg_file = script_dir / "social-card.svg"
    
    # Check if SVG exists
    if not svg_file.exists():
        print(f"❌ Error: {svg_file} not found!")
        return False
    
    print("🎨 AutoML Social Card PNG Generator")
    print("=" * 50)
    print(f"📂 Working directory: {script_dir}")
    print(f"📄 Source file: {svg_file.name}")
    print()
    
    # Try to import cairosvg
    try:
        import cairosvg
        use_cairo = True
        print("✅ Using CairoSVG for conversion (best quality)")
    except (ImportError, OSError) as e:
        use_cairo = False
        if "cairo" in str(e).lower():
            print("⚠️  Cairo library not found (common on Windows)")
            print("\n📌 RECOMMENDED: Use the HTML generator instead:")
            print("   1. Open 'generate-social-card.html' in your browser")
            print("   2. Click 'Download PNG (1280×640)'")
            print("   3. Upload to GitHub → Settings → Social preview")
            print("\n   This method works perfectly and requires no installation!")
            print("\nAlternatively, install Cairo:")
            print("   • Windows: Download GTK+ from https://gtk.org or use WSL")
            print("   • Linux: sudo apt-get install libcairo2-dev")
            print("   • Mac: brew install cairo")
            return False
        else:
            print("⚠️  CairoSVG not found. Trying alternative method...")
            print("   For best results, install: pip install cairosvg")
    
    # Output files
    outputs = [
        {
            "name": "social-card.png",
            "width": 1280,
            "height": 640,
            "scale": 1,
            "description": "Standard resolution"
        },
        {
            "name": "social-card@2x.png",
            "width": 2560,
            "height": 1280,
            "scale": 2,
            "description": "High resolution (Retina)"
        }
    ]
    
    if use_cairo:
        # Convert using CairoSVG (best quality)
        for output in outputs:
            output_path = script_dir / output["name"]
            
            print(f"\n🖼️  Generating {output['name']}...")
            print(f"   Size: {output['width']}×{output['height']}")
            print(f"   {output['description']}")
            
            try:
                cairosvg.svg2png(
                    url=str(svg_file),
                    write_to=str(output_path),
                    output_width=output['width'],
                    output_height=output['height'],
                    dpi=96 * output['scale']
                )
                
                file_size = output_path.stat().st_size / 1024
                print(f"   ✅ Success! ({file_size:.1f} KB)")
                
            except Exception as e:
                print(f"   ❌ Error: {e}")
                return False
    
    else:
        # Try alternative method using PIL and svglib
        try:
            from svglib.svglib import svg2rlg
            from reportlab.graphics import renderPM
            from PIL import Image
            
            print("✅ Using svglib + ReportLab for conversion")
            
            for output in outputs:
                output_path = script_dir / output["name"]
                
                print(f"\n🖼️  Generating {output['name']}...")
                print(f"   Size: {output['width']}×{output['height']}")
                print(f"   {output['description']}")
                
                try:
                    # Convert SVG to RLG drawing
                    drawing = svg2rlg(str(svg_file))
                    
                    # Scale the drawing
                    scale = output['scale']
                    drawing.width = output['width']
                    drawing.height = output['height']
                    drawing.scale(scale, scale)
                    
                    # Render to PNG
                    renderPM.drawToFile(
                        drawing,
                        str(output_path),
                        fmt='PNG',
                        dpi=96 * scale
                    )
                    
                    file_size = output_path.stat().st_size / 1024
                    print(f"   ✅ Success! ({file_size:.1f} KB)")
                    
                except Exception as e:
                    print(f"   ❌ Error: {e}")
                    return False
                    
        except ImportError:
            print("\n❌ No suitable conversion library found!")
            print("\nPlease install one of the following:")
            print("  Option 1 (Recommended): pip install cairosvg")
            print("  Option 2: pip install svglib reportlab pillow")
            print("\nOr use the generate-social-card.html file in your browser.")
            return False
    
    print("\n" + "=" * 50)
    print("✨ Conversion complete!")
    print("\n📊 Generated files:")
    for output in outputs:
        output_path = script_dir / output["name"]
        if output_path.exists():
            print(f"   ✓ {output['name']} ({output['width']}×{output['height']})")
    
    print("\n🚀 Next steps:")
    print("   1. Go to your GitHub repository settings")
    print("   2. Navigate to 'Social preview' section")
    print("   3. Upload social-card.png")
    print("   4. Save and verify the preview appears when sharing!")
    
    return True


def main():
    """Main entry point."""
    try:
        success = convert_svg_to_png()
        exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\n⚠️  Conversion cancelled by user")
        exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        exit(1)


if __name__ == "__main__":
    main()
