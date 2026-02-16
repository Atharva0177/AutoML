# AutoML Social Card Assets

This folder contains the social media card (repository preview image) for the AutoML project.

## ⚡ Quick Start (Windows)

**Double-click** `generate-social-card.bat` or `generate-social-card.html` to get started!

## 📁 Files

- **`social-card.svg`** - Scalable vector graphics version (editable)
- **`generate-social-card.html`** - Interactive HTML tool to preview and download the social card
- **`generate-social-card.bat`** - Windows launcher (double-click to open HTML tool)
- **`convert-to-png.py`** - Python script to convert SVG to PNG format (Linux/Mac)
- **`README.md`** - This documentation file

## 🎨 Social Card Features

The social card highlights the key features of AutoML:
- **27+ Models**: 18 Traditional ML + 9+ Deep Learning models
- **GPU Accelerated**: Support for XGBoost, LightGBM, CatBoost, PyTorch
- **Multi-Modal**: Tabular, Image, and Text data processing
- **Smart AI Recommender**: Intelligent model selection system
- **Advanced Capabilities**: Preprocessing, feature engineering, optimization
- **Production Ready**: 317 passing tests, MLflow tracking, Optuna optimization

## 📐 Specifications

- **Dimensions**: 1280 × 640 pixels
- **Aspect Ratio**: 2:1 (GitHub recommended)
- **Safe Border**: 40px around all edges
- **Format**: SVG (vector) and PNG (raster)
- **Color Scheme**: Purple gradient (#667eea → #764ba2 → #f093fb)

## 🚀 How to Use

### Method 1: Using the HTML Generator ⭐ (Recommended for Windows)

**No installation required! Works in any browser.**

1. **Open** `generate-social-card.html` in your web browser (double-click the file)
2. **Preview** the social card to see how it looks
3. **Click** "Download PNG (1280×640)" to download the standard version
4. **Optional**: Click "Download PNG @2x" for high-resolution version (2560×1280)

✅ This is the **easiest and most reliable method**, especially on Windows!

### Method 2: Using Python Script (Linux/Mac or with Cairo installed)

```bash
# Install required packages
pip install cairosvg pillow

# Run the conversion script
python convert-to-png.py
```

**Note for Windows users:** This requires installing GTK+ and Cairo libraries, which can be complex. We recommend using Method 1 (HTML) instead.

This will generate:
- `social-card.png` (1280×640)
- `social-card@2x.png` (2560×1280)

### Method 3: Online SVG to PNG Converter

1. Upload `social-card.svg` to any online converter:
   - [CloudConvert](https://cloudconvert.com/svg-to-png)
   - [SVG to PNG Online](https://svgtopng.com/)
   - [Convertio](https://convertio.co/svg-png/)
2. Set dimensions to 1280×640
3. Download the PNG file

## 📢 Setting Up on GitHub

### Repository Social Preview

1. Go to your GitHub repository
2. Click **Settings** (top menu)
3. Scroll down to **Social preview** section
4. Click **Edit**
5. Upload `social-card.png` (the PNG version)
6. Click **Save**

Now when anyone shares your repository link on:
- Twitter/X
- LinkedIn
- Facebook
- Discord
- Slack
- Any platform supporting Open Graph

...they will see this beautiful preview card! 🎉

### Verify It's Working

After uploading:
1. Go to [Twitter Card Validator](https://cards-dev.twitter.com/validator)
2. Enter your repository URL
3. See the preview

Or use [LinkedIn Post Inspector](https://www.linkedin.com/post-inspector/)

## ✏️ Customizing the Card

The SVG file can be edited with:
- **Vector Graphics Software**: Adobe Illustrator, Inkscape, Figma
- **Code Editor**: VS Code, Sublime Text (it's just XML)
- **Online Editors**: [Boxy SVG](https://boxy-svg.com/), [SVG-Edit](https://svg-edit.github.io/)

### Quick Edits in Code

Open `social-card.svg` in any text editor and modify:

```xml
<!-- Change the title -->
<text x="240" y="210" ...>
    AutoML  <!-- Edit this -->
</text>

<!-- Change the subtitle -->
<text x="240" y="255" ...>
    Advanced Automated Machine Learning System  <!-- Edit this -->
</text>

<!-- Change colors in gradient -->
<linearGradient id="bgGradient" ...>
    <stop offset="0%" style="stop-color:#667eea;..."/>  <!-- Edit color -->
    ...
</linearGradient>
```

### Color Palette Used

```css
Primary Gradient:
  - Start: #667eea (Purple Blue)
  - Middle: #764ba2 (Deep Purple)
  - End: #f093fb (Light Pink)

Accent Colors:
  - Success: #11998e (Teal)
  - White: #ffffff (Text & Icons)
  - Transparent Overlays: rgba(255,255,255,0.05-0.25)
```

## 📊 Design Principles

1. **40px Safe Border**: All important content is 40px from edges to prevent cropping
2. **High Contrast**: White text on gradient background for readability
3. **Clear Hierarchy**: Large title, descriptive subtitle, feature badges, details
4. **Visual Balance**: Icons and text balanced across the card
5. **Brand Colors**: Professional purple gradient consistent with ML/AI branding
6. **Feature Highlights**: Key capabilities visible at a glance

## 🔧 Troubleshooting

### Cairo Library Error on Windows

If you see errors like "no library called 'cairo-2' was found":

**✅ Solution:** Use the **HTML generator** (`generate-social-card.html`) instead! It works perfectly without any installations.

Alternatively, you can:
- Use WSL (Windows Subsystem for Linux) and install Cairo there
- Download GTK+ for Windows (complex, not recommended)
- Use an online converter

### Image Not Showing on GitHub

- Wait 5-10 minutes for GitHub to process the image
- Clear your browser cache
- Make sure file size is under 1MB
- Ensure dimensions are exactly 1280×640

### SVG Not Converting Properly

- Use the HTML generator (most reliable)
- Or install CairoSVG: `pip install cairosvg pillow`
- Or use Inkscape: `inkscape social-card.svg --export-png=social-card.png --export-width=1280`

### Text Looks Blurry

- Use @2x version (2560×1280) and let platforms scale down
- Ensure you're exporting as PNG, not compressing as JPEG
- Use the HTML canvas method for crisp text rendering

## 📝 License

This social card is part of the AutoML project and follows the same license.

---

**Need help?** Open an issue in the main repository or check the [README.md](../README.md) for more information.
