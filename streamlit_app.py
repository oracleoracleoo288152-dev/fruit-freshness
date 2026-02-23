import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
import time
import re
import difflib
from db import save_upload
import os

# ---------------- PAGE SETUP ----------------
st.set_page_config(page_title="Rotten or Not 🍎", layout="wide")

# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_model():
    return YOLO("best1.pt")

model = load_model()


# --- Multilingual support ---
LANG_OPTIONS = {"English": "en", "हिन्दी": "hi", "ગુજરાતી": "gu"}
lang_choice = st.selectbox("Language / भाषा / ભાષા", list(LANG_OPTIONS.keys()), index=0)
LANG = LANG_OPTIONS.get(lang_choice, "en")

TRANSLATIONS = {
    "app_title": {
        "en": "🍓 Fruit Freshness Detector",
        "hi": "🍓 फल ताज़गी डिटेक्टर",
        "gu": "🍓 ફળ તાજગી ડિટેક્ટર"
    },
    "app_subtitle": {
        "en": "Detect whether a fruit is **fresh** or **rotten** using YOLO",
        "hi": "YOLO का उपयोग करके पता करें कि फल ताज़ा है या सड़ा हुआ",
        "gu": "YOLO નો ઉપયોગ કરીને ફળ તाजा છે કે સુંકી ગયું છે તે શોધો"
    },
    "upload_header": {
        "en": "📤 Upload Fruit Image",
        "hi": "📤 फल की तस्वीर अपलोड करें",
        "gu": "📤 ફળની છબી અપલોડ કરો"
    },
    "upload_label": {
        "en": "Upload Image",
        "hi": "इमेज अपलोड करें",
        "gu": "ચિત્ર અપલોડ કરો"
    },
    "uploaded_caption": {
        "en": "Uploaded Image",
        "hi": "अपलोड की गई इमेज",
        "gu": "અપલોડ કરેલ છબી"
    },
    "detection_caption": {
        "en": "Detection Result",
        "hi": "डिटेक्शन परिणाम",
        "gu": "ડિટેક્શન પરિણામ"
    },
    "no_fruit": {
        "en": "⚠️ No fruit detected.",
        "hi": "⚠️ कोई फल नहीं मिला.",
        "gu": "⚠️ કોઈ ફળ શોધાયું નથી."
    },
    "webcam_header": {
        "en": "🎥 Live Webcam Detection",
        "hi": "🎥 लाइव वेबकैम डिटेक्शन",
        "gu": "🎥 લાઈવ વેબકેમ ડિટેક્શન"
    },
    "start_webcam": {
        "en": "Start Webcam",
        "hi": "वेबकैम शुरू करें",
        "gu": "વેબકેમ શરૂ કરો"
    },
    "stop_webcam": {
        "en": "Stop Webcam",
        "hi": "वेबकैम रोकें",
        "gu": "વેબકેમ બંધ કરો"
    },
    "webcam_stopped": {
        "en": "🛑 Webcam stopped.",
        "hi": "🛑 वेबकैम रुकी।",
        "gu": "🛑 વેબકેમ બંધ થઈ ગઈ."
    },
    "camera_error": {
        "en": "Camera error.",
        "hi": "कैमरा त्रुटि.",
        "gu": "કેમેરા ભૂલ."
    },
    "recipes_header": {
        "en": "Recipe Ideas",
        "hi": "रेसिपी सुझाव",
        "gu": "રીસપી વિચારો"
    },
    "no_recipe_for": {
        "en": "No recipe found for {name}.",
        "hi": "{name} के लिए कोई रेसिपी नहीं मिली.",
        "gu": "{name} માટે રેસપી મળી નથી."
    },
    "model_loaded": {
        "en": "✅ Model loaded successfully!",
        "hi": "✅ मॉडल सफलतापूर्वक लोड हुआ!",
        "gu": "✅ મોડલ સફળતાપૂર્વક લોડ થયું!"
    }
    ,
    "detection_details": {
        "en": "Detection details",
        "hi": "डिटेक्शन विवरण",
        "gu": "ડિટેક્શન વિગતો"
    },
    "select_recipe": {
        "en": "Select fruit for recipe (override)",
        "hi": "रेसिपी के लिए फल चुनें (ओवरराइड)",
        "gu": "રીસપી માટે ફળ પસંદ કરો (ઓવરરાઈડ)"
    }
    ,
    "auto_map": {
        "en": "Auto-select best match",
        "hi": "सबसे अच्छा मेल स्वचालित रूप से चुनें",
        "gu": "સરસ મૅચ આપમેળે પસંદ કરો"
    },
    "confidence_threshold": {
        "en": "Confidence threshold",
        "hi": "विश्वास सीमा",
        "gu": "વિશ્વાસ થ્રેશોલ્ડ"
    },
    "auto_map_info": {
        "en": "Auto-mapping uses label normalization, substring and fuzzy match.",
        "hi": "ऑटो-मैपिंग लेबल सामान्यीकरण, सबस्ट्रिंग और फजी मिलान का उपयोग करता है।",
        "gu": "આપમેળે મેપિંગ લેબલ નોર્મલાઈઝેશન, સબસ્ટ્રિંગ અને ફઝી મેચનો ઉપયોગ કરે છે."
    },
    "auto_map_failed": {
        "en": "Auto-mapping couldn't find a good match; please select manually.",
        "hi": "ऑटो-मैपिंग में अच्छा मेल नहीं मिला; कृपया मैन्युअली चुनें।",
        "gu": "આપમેળે શોધી શક્યું નહી; કૃપા કરી મેન્યુઅલી પસંદ કરો."
    }
}

def t(key, **kwargs):
    entry = TRANSLATIONS.get(key, {})
    text = entry.get(LANG, entry.get("en", ""))
    if kwargs:
        try:
            return text.format(**kwargs)
        except Exception:
            return text
    return text

# show translated title/subtitle
st.title(t("app_title"))
st.markdown(t("app_subtitle"))
st.success(t("model_loaded"))

# Simple recipe database (extend as needed)
RECIPES = {
    "apple": {
        "title": "Apple Crumble",
        "content": "Ingredients:\n- 4 apples\n- 100g flour\n- 75g butter\n- 75g brown sugar\n\nSteps:\n1. Slice apples and place in a baking dish.\n2. Mix flour, butter and sugar into crumbs and sprinkle over apples.\n3. Bake at 180°C for 30-35 minutes until golden."
    },
    "banana": {
        "title": "Banana Smoothie",
        "content": "Ingredients:\n- 2 ripe bananas\n- 250ml milk (or plant milk)\n- 1 tbsp honey\n\nSteps:\n1. Blend all ingredients until smooth.\n2. Serve chilled."
    },
    "mango": {
        "title": "Mango Salsa",
        "content": "Ingredients:\n- 1 ripe mango\n- 1/2 red onion\n- Juice of 1 lime\n- Handful cilantro\n\nSteps:\n1. Dice mango and onion.\n2. Mix with lime juice and chopped cilantro.\n3. Serve with chips or grilled fish."
    },
    "orange": {
        "title": "Orange Granita",
        "content": "Ingredients:\n- 500ml fresh orange juice\n- 50g sugar\n\nSteps:\n1. Dissolve sugar into juice.\n2. Freeze in a shallow tray, scraping every 30 minutes until flaky."
    },
    "strawberry": {
        "title": "Strawberry Salad",
        "content": "Ingredients:\n- 250g strawberries\n- Handful of spinach\n- Balsamic vinaigrette\n\nSteps:\n1. Halve strawberries and toss with spinach.\n2. Drizzle with vinaigrette and serve."
    }
    ,
    "cucumber": {
        "title": "Cucumber Raita",
        "content": "Ingredients:\n- 1 large cucumber\n- 250g plain yogurt\n- 1/2 tsp roasted cumin powder\n- Salt to taste\n- Fresh cilantro or mint (optional)\n\nSteps:\n1. Peel and grate or finely chop the cucumber.\n2. Mix cucumber with yogurt, cumin powder and salt.\n3. Garnish with chopped cilantro or mint and serve chilled as a side."
    }
}

# Translations for recipes (Hindi and Gujarati)
RECIPES_TRANSLATIONS = {
    "hi": {
        "apple": {
            "title": "एप्पल क्रम्बल",
            "content": "सामग्री:\n- 4 सेब\n- 100g मैदा\n- 75g मक्खन\n- 75g ब्राउन शुगर\n\nविधि:\n1. सेब काटकर बेकिंग डिश में रखें।\n2. मैदा, मक्खन और शुगर मिलाकर क्रम्बल बनाकर सेब पर छिड़कें।\n3. 180°C पर 30-35 मिनट बेक करें।"
        },
        "banana": {
            "title": "केला स्मूदी",
            "content": "सामग्री:\n- 2 पके केले\n- 250ml दूध (या प्लांट-मिल्क)\n- 1 बड़ा चम्मच शहद\n\nविधि:\n1. सभी सामग्री ब्लेंड करें।\n2. ठंडा परोसें।"
        },
        "mango": {
            "title": "मैंगो सालसा",
            "content": "सामग्री:\n- 1 पका आम\n- 1/2 लाल प्याज\n- 1 नींबू का रस\n- थोड़ी हरी धनिया\n\nविधि:\n1. आम और प्याज को काटें।\n2. नींबू का रस और धनिया मिलाकर परोसें।"
        },
        "orange": {
            "title": "संतरे की ग्रैनिटा",
            "content": "सामग्री:\n- 500ml ताजा संतरे का रस\n- 50g चीनी\n\nविधि:\n1. चीनी घोलकर रस में मिलाएं।\n2. एक शैलो ट्रे में फ्रीज करें और हर 30 मिनट में खुरचें जब तक फलेक जैसा ना हो।"
        },
        "strawberry": {
            "title": "स्ट्रॉबेरी सलाद",
            "content": "सामग्री:\n- 250g स्ट्रॉबेरी\n- कुछ पालक\n- बेलसामिक विनेग्रेट\n\nविधि:\n1. स्ट्रॉबेरी आधी करें और पालक के साथ मिलाएं।\n2. विनेग्रेट डालें और परोसें।"
        },
        "cucumber": {
            "title": "खीरे की रायता",
            "content": "सामग्री:\n- 1 बड़ा खीरा\n- 250g दही\n- 1/2 चम्मच भुना जीरा पाउडर\n- स्वादानुसार नमक\n- हरा धनिया या पुदीना\n\nविधि:\n1. खीरा कद्दूकस या बारीक काटें।\n2. दही में मिलाकर मसाले डालें और ठंडा परोसें।"
        }
    },
    "gu": {
        "apple": {
            "title": "એપલ ક્રંબલ",
            "content": "સામગ્રી:\n- 4 સફરજન\n- 100g મેંદો\n- 75g માખણ\n- 75g બ્રાઉન ખાંડ\n\nરીત:\n1. સફરજન કાપીને બેકિંગ ડિશમાં મૂકો.\n2. મેંદો, માખણ અને ખાંડ મિક્સ કરીને છાંટો.\n3. 180°C પર 30-35 મિનિટ બેક કરો."
        },
        "banana": {
            "title": "બનાના સ્મૂદી",
            "content": "સામગ્રી:\n- 2 પેલા કેળા\n- 250ml દૂધ (અથવા પ્લાન્ટ મિલ્ક)\n- 1 વડી ચમચી મધ\n\nરીત:\n1. તમામ સામગ્રી બ્લેન્ડ કરો.\n2. ઠંડુ પરોછો."
        },
        "mango": {
            "title": "કેરી સલસા",
            "content": "સામગ્રી:\n- 1 પકડેલ કેરી\n- 1/2 લાલ ડુંગળી\n- 1 લાઇમ નો રસ\n- થોડું ધ્નિયાનો પત્તો\n\nરીત:\n1. કેરી અને ડુંગળી કાપો.\n2. લાઇમ રસ અને ધ્નિયા સાથે મિક્સ કરો."
        },
        "orange": {
            "title": "સંટારા ગ્રાનિતા",
            "content": "સામગ્રી:\n- 500ml તાજું સંતરાનો રસ\n- 50g ખાંડ\n\nરીત:\n1. ખાંડ ગળાવો અને રસમાં મિક્સ કરો.\n2. પટલા ટ્રેમાં ફ્રીઝ કરો અને દર 30 મિનિટે ખુરચો."
        },
        "strawberry": {
            "title": "સ્ટ્રોબેરી સલાડ",
            "content": "સામગ્રી:\n- 250g સ્ટ્રોબેરી\n- થોડો સ્પિનેચ\n- બેલસાયમિક વિનેગ્રેટ\n\nરીત:\n1. સ્ટ્રોબેરી કાપી સ્પિનેચ સાથે મિક્સ કરો.\n2. વિનેગ્રેટ ઉમેરો અને સર્વ કરો."
        },
        "cucumber": {
            "title": "કાકડીનું રાયতা",
            "content": "સામગ્રી:\n- 1 મોટી કાકડી\n- 250g દહીં\n- 1/2 ચમચી ભુનો જીરુ પાવડર\n- સ્વાદ માટે મીઠું\n- ધનિયા અથવા પુદીના પત્તા\n\nરીત:\n1. કાકડી છીલીને કાપો અથવા કુરજુ કરો.\n2. દહીંમાં મિક્સ કરો અને મસાલા ઉમેરો. ઠંડું સર્વ કરો."
        }
    }
}

def extract_fruit_name(label: str) -> str:
    """Normalize model label to a fruit name key used in RECIPES."""
    s = label.lower()
    s = s.replace("_", " ")
    # remove words indicating freshness
    s = re.sub(r"\b(fresh|rotten|ripe|unripe|good|bad)\b", "", s)
    s = re.sub(r"[^a-z\s]", "", s)
    s = s.strip()
    # if label contains multiple words, pick the last as likely fruit (common model patterns)
    parts = s.split()
    if len(parts) == 0:
        return ""
    # try to find a known fruit in parts
    for p in parts:
        if p in RECIPES:
            return p
    # fallback to last token
    return parts[-1]


def auto_map_fruit(detected_info, conf_thresh=0.3):
    """Try to auto-map model detections to a known recipe key.

    Strategy (in order of checking per detection sorted by confidence):
    - Normalize label and check exact recipe key
    - Check if any recipe key is substring of label
    - Fuzzy match label against recipe keys using difflib
    Returns the first reasonable match or None.
    """
    if not detected_info:
        return None

    # sort by confidence desc
    items = sorted(detected_info, key=lambda x: x.get("conf", 0), reverse=True)
    keys = list(RECIPES.keys())

    for it in items:
        conf = float(it.get("conf", 0))
        if conf < conf_thresh:
            continue
        label = it.get("label", "").lower()
        name = extract_fruit_name(label)
        if name in RECIPES:
            return name
        # substring
        for k in keys:
            if k in label:
                return k
        # fuzzy match against full label
        match = difflib.get_close_matches(label, keys, n=1, cutoff=0.6)
        if match:
            return match[0]
        # try tokens
        for token in label.split():
            match = difflib.get_close_matches(token, keys, n=1, cutoff=0.7)
            if match:
                return match[0]

    return None

# ===
# =====================================================
st.header(t("upload_header"))

uploaded_file = st.file_uploader(
    t("upload_label"),
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    # read raw bytes once so we can both decode and save them
    raw_bytes = uploaded_file.read()
    file_bytes = np.asarray(bytearray(raw_bytes), dtype=np.uint8)
    frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # resize large images
    frame_resized = cv2.resize(frame_rgb, (640, 640))

    st.image(frame_rgb, caption=t("uploaded_caption"), width="stretch")

    results = model.predict(frame_resized, conf=0.5, verbose=False)
    pred = results[0]

    if pred.boxes is not None and len(pred.boxes) > 0:
        detected_labels = []
        detected_info = []
        for box in pred.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            label = pred.names[cls_id]
            detected_labels.append(label)
            detected_info.append({"label": label, "conf": float(conf), "cls_id": int(cls_id)})

            color = (0,255,0) if "fresh" in label.lower() else (0,0,255)

            cv2.rectangle(frame_resized,(x1,y1),(x2,y2),color,2)
            cv2.putText(frame_resized,
                        f"{label} {conf:.2f}",
                        (x1,y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        color,
                        2)

        st.image(frame_resized,
             caption=t("detection_caption"),
             width="stretch")

        # Show detection details and allow manual override for recipe selection
        st.markdown("---")
        if t("detection_details", ):
            pass
        with st.expander(t("detection_details")):
            st.write("Detected labels and confidences:")
            st.write(detected_info)
            st.write("Model class mapping (id -> name):")
            try:
                st.write(pred.names)
            except Exception:
                st.write("(no mapping available)")

        # Auto-mapping controls
        st.markdown(":information_source: " + t("auto_map_info"))
        col1, col2 = st.columns([1, 2])
        with col1:
            auto = st.checkbox(t("auto_map"), value=True)
        with col2:
            conf_thresh = st.slider(t("confidence_threshold"), 0.0, 1.0, 0.3, 0.05)

        options = sorted(RECIPES.keys())
        chosen_fruit = None
        if auto:
            auto_choice = auto_map_fruit(detected_info, conf_thresh=conf_thresh)
            if auto_choice:
                chosen_fruit = auto_choice
                st.success(f"Auto-selected: {chosen_fruit}")
            else:
                st.warning(t("auto_map_failed"))

        # If not auto-selected, show manual selector (default to first detected normalized)
        if not chosen_fruit:
            # Build default selection (first normalized detected fruit if any)
            fruit_keys = []
            for lab in detected_labels:
                name = extract_fruit_name(lab)
                if name:
                    fruit_keys.append(name)

            default_idx = 0
            if len(fruit_keys) > 0 and fruit_keys[0] in options:
                default_idx = options.index(fruit_keys[0])

            chosen_fruit = st.selectbox(t("select_recipe"), options, index=default_idx)

        st.header(t("recipes_header"))
        if LANG in RECIPES_TRANSLATIONS and chosen_fruit in RECIPES_TRANSLATIONS[LANG]:
            r = RECIPES_TRANSLATIONS[LANG][chosen_fruit]
            st.subheader(r.get("title", chosen_fruit.title()))
            st.text(r.get("content", ""))
        elif chosen_fruit in RECIPES:
            r = RECIPES[chosen_fruit]
            st.subheader(r.get("title", chosen_fruit.title()))
            st.text(r.get("content", ""))
        else:
            st.info(t("no_recipe_for", name=chosen_fruit))

        # Try saving upload + detection metadata to MongoDB and Cloudinary (non-fatal)
        try:
            cloud_cfg = {
                "cloud_name": os.getenv("CLOUDINARY_CLOUD_NAME", "dgosjbdx7"),
                "api_key": os.getenv("CLOUDINARY_API_KEY", "764318225397556"),
                "api_secret": os.getenv("CLOUDINARY_API_SECRET", "2_tKwqV7ZpG0d-nfgADM6jBXHnQ"),
            }
            save_res = save_upload(raw_bytes, getattr(uploaded_file, "name", "upload"), chosen_fruit, detected_info, cloudinary_config=cloud_cfg)
            st.caption(f"Saved upload to database: {str(save_res.get('_id'))}")
            if save_res.get("cloudinary"):
                st.markdown(f"Uploaded to Cloudinary: {save_res['cloudinary'].get('secure_url')}")
        except Exception as e:
            st.warning(f"Could not save upload to database/cloud: {e}")

    else:
        st.warning("⚠️ No fruit detected.")


# =====================================================
# 🎥 WEBCAM DETECTION
# =====================================================
st.header(t("webcam_header"))

start_detection = st.button(t("start_webcam"))
FRAME_WINDOW = st.image([], width="stretch")

if start_detection:
    cap = cv2.VideoCapture(0)
    stop_button = st.button(t("stop_webcam"))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            st.error(t("camera_error"))
            break

        frame = cv2.flip(frame, 1)

        results = model.predict(frame, conf=0.5, verbose=False)
        pred = results[0]

        if pred.boxes is not None and len(pred.boxes) > 0:
            for box in pred.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                label = pred.names[cls_id]

                color = (0,255,0) if "fresh" in label.lower() else (0,0,255)

                cv2.rectangle(frame,(x1,y1),(x2,y2),color,2)
                cv2.putText(frame,
                            f"{label} {conf:.2f}",
                            (x1,y1-10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.8,
                            color,
                            2)

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        FRAME_WINDOW.image(frame, width="stretch")

        if stop_button:
            break

        time.sleep(0.03)

    cap.release()
    st.warning(t("webcam_stopped"))
