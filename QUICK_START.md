# Quick Start Guide 🚀

Get up and running with the Advanced Drowsiness Detection System in 5 minutes!

## Prerequisites

- Python 3.8 or higher
- Webcam (built-in or USB)
- Modern web browser (Chrome, Firefox, Edge)

## Installation Steps

### 1. Install Dependencies

```bash
# Navigate to project directory
cd drowsy

# Install required packages
pip install -r requirements.txt
```

**Alternative (if issues occur):**
```bash
# Run automated setup
python setup.py
```

### 2. Verify Installation

```bash
# Quick verification
python -c "import cv2, mediapipe; print('Installation successful!')"
```

If you see "Installation successful!" you're ready to go!

## First Run

### Step 1: Start the System

```bash
python app.py
```

You should see:
```
Initializing Advanced Drowsiness Detection System...
System initialized successfully!
Starting web server on http://0.0.0.0:5000
```

### Step 2: Open Dashboard

Open your web browser and navigate to:
```
http://localhost:5000
```

### Step 3: Calibrate (First Time Only)

**This is IMPORTANT for accurate detection!**

1. Click the **"Calibrate"** button
2. Sit naturally at your normal position
3. Look at the camera with a neutral expression
4. Wait 30 seconds for calibration to complete
5. You'll see a success message when done

**During calibration:**
- ✅ DO: Sit naturally, blink normally
- ❌ DON'T: Move around, close eyes intentionally, yawn

### Step 4: Start Detection

1. Click **"Start Detection"**
2. The video feed will appear
3. Metrics will update in real-time
4. You're now being monitored!

## Understanding the Dashboard

### Metrics Panel (Right Side)

**Fatigue Score** (0-100%)
- 0-30%: Normal (green)
- 30-50%: Early warning (yellow)
- 50-70%: Moderate fatigue (orange)
- 70-85%: High fatigue (dark orange)
- 85-100%: Critical drowsiness (red)

**Key Indicators:**
- **EAR**: Eye Aspect Ratio (lower = more closed)
- **PERCLOS**: % of time eyes are closed
- **Blinks/Yawns**: Count and rate per minute
- **Microsleeps**: Brief sleep episodes detected

### Alert System

The system uses 5 alert levels:

| Level | Color | Meaning | Action |
|-------|-------|---------|--------|
| 0 | Green | Normal | Continue |
| 1 | Yellow | Early Warning | Be aware |
| 2 | Orange | Moderate Fatigue | Consider break |
| 3 | Dark Orange | High Fatigue | Take break soon |
| 4 | Red | Critical | STOP IMMEDIATELY |

**Alert Sounds:**
- Level 1: Single gentle beep
- Level 2: Double beep
- Level 3: Rising tone
- Level 4: Urgent pulsing alarm

### Charts

**Fatigue Trend (Bottom Left)**
- Shows last 30 seconds of fatigue score
- Rising trend = increasing fatigue
- Spikes indicate sudden drowsiness

**EAR & PERCLOS (Bottom Right)**
- Blue line: Eye Aspect Ratio
- Orange line: Eye closure percentage
- Helps identify patterns

## Common Usage Scenarios

### For Drivers 🚗

**Setup:**
1. Mount phone/tablet on dashboard
2. Run system on laptop/tablet
3. Ensure camera has clear view of face

**Best Practices:**
- Calibrate while feeling alert
- Take breaks if Level 3 alerts occur
- STOP if Level 4 alerts appear
- Don't rely solely on this system - get adequate sleep!

### For Students/Office Workers 💻

**Setup:**
1. Position webcam at eye level
2. Ensure good lighting from front/sides
3. Avoid backlighting (window behind you)

**Best Practices:**
- Use during long study/work sessions
- Take breaks every hour regardless
- Pay attention to Level 2+ alerts
- Review session analytics to improve habits

### For Research 🔬

**Setup:**
1. Use high-quality webcam (720p+)
2. Controlled lighting environment
3. Configure `config.yaml` for your needs

**Data Collection:**
- Sessions auto-saved to `sessions/` folder
- Export to CSV for analysis
- Use `session_analyzer.py` for ML insights

## Stopping a Session

1. Click **"Stop Detection"**
2. Session data automatically saved
3. View statistics in dashboard
4. Analyze session using `session_analyzer.py`

## Session Analysis

After stopping:

```bash
# Generate detailed report
python session_analyzer.py sessions/session_20250131_143022.json txt

# Or export as JSON
python session_analyzer.py sessions/session_20250131_143022.json json
```

The report includes:
- Risk assessment
- Session statistics
- Alert summary
- Personalized recommendations

## Customization

### Adjust Sensitivity

Edit `config.yaml`:

```yaml
# Make detection MORE sensitive (more alerts)
thresholds:
  ear_baseline: 0.27  # Default: 0.25
  perclos_warning: 0.12  # Default: 0.15

# Make detection LESS sensitive (fewer alerts)
thresholds:
  ear_baseline: 0.23
  perclos_warning: 0.20
```

### Change Alert Sounds

```yaml
audio:
  enabled: true     # Set to false to disable
  volume: 0.5       # Range: 0.0 - 1.0
```

### Modify Camera Settings

```yaml
camera:
  id: 0            # Try 1, 2, etc. if default doesn't work
  width: 1280      # Lower for faster processing
  height: 720
```

## Troubleshooting

### Camera Not Working

**Problem:** "Could not open camera"

**Solutions:**
1. Check camera permissions
2. Close other apps using camera (Zoom, Skype, etc.)
3. Try different camera ID in `config.yaml`
4. Test camera: `python -c "import cv2; print(cv2.VideoCapture(0).isOpened())"`

### Low FPS

**Problem:** Video is laggy or slow

**Solutions:**
1. Reduce camera resolution in `config.yaml`
2. Close other resource-heavy applications
3. Ensure good lighting (reduces processing)

### False Alerts

**Problem:** Too many alerts when feeling alert

**Solutions:**
1. **Re-calibrate** (most common fix)
2. Increase thresholds in `config.yaml`
3. Ensure proper lighting
4. Check camera angle (should see full face)

### No Alerts

**Problem:** No alerts even when tired

**Solutions:**
1. Check if calibration was done correctly
2. Decrease thresholds in `config.yaml`
3. Verify face is clearly visible
4. Check alert cooldown settings

### Port Already in Use

**Problem:** "Address already in use"

**Solution:**
```yaml
# In config.yaml
dashboard:
  port: 5001  # Change to different port
```

## Tips for Best Results

✅ **DO:**
- Calibrate when well-rested
- Ensure good, even lighting
- Keep face centered in frame
- Use in combination with breaks
- Review session analytics

❌ **DON'T:**
- Rely solely on system (get adequate sleep!)
- Calibrate when tired
- Use with sunglasses
- Position camera too far/close
- Ignore Level 3+ alerts

## System Requirements Check

Minimum specs:
- ✅ Python 3.8+
- ✅ 4GB RAM
- ✅ Webcam (any quality)
- ✅ Modern browser

Recommended:
- ⭐ Python 3.10+
- ⭐ 8GB RAM
- ⭐ 720p+ webcam
- ⭐ Good lighting

## What's Next?

Once you're comfortable with the basics:

1. **Review README.md** - Comprehensive documentation
2. **Explore config.yaml** - Advanced customization
3. **Analyze sessions** - Learn from your patterns
4. **Adjust thresholds** - Personalize detection

## Support

If you encounter issues:
1. Check this guide first
2. Review README.md
3. Verify camera permissions
4. Run `python setup.py` again
5. Check `config.yaml` settings

## Summary

```bash
# Complete workflow
cd drowsy
pip install -r requirements.txt
python app.py
# Open http://localhost:5000
# Click "Calibrate" → Wait 30s → "Start Detection"
```

**You're all set! Stay alert and safe! 🚗💤**
