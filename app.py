import pandas as pd
import joblib
import ipaddress
from urllib.parse import urlparse
from flask import Flask, request, render_template_string

app = Flask(__name__)

# Load trained phishing detection model
bundle = joblib.load("phishing_model.joblib")
model = bundle["model"]
feature_names = bundle["feature_names"]
model_name = bundle["model_name"]


def extract_url_features(url):
    if not url.startswith(("http://", "https://")):
        url = "http://" + url

    parsed = urlparse(url)
    hostname = parsed.hostname or ""

    suspicious_words = [
        "login", "verify", "update", "secure",
        "account", "bank", "password", "signin",
        "confirm", "wallet"
    ]

    try:
        ipaddress.ip_address(hostname)
        has_ip = 1
    except ValueError:
        has_ip = 0

    try:
        has_port = int(parsed.port is not None)
    except ValueError:
        has_port = 0

    return {
        "url_length": len(url),
        "hostname_length": len(hostname),
        "path_length": len(parsed.path),
        "query_length": len(parsed.query),
        "dot_count": url.count("."),
        "hyphen_count": url.count("-"),
        "at_count": url.count("@"),
        "question_count": url.count("?"),
        "equal_count": url.count("="),
        "ampersand_count": url.count("&"),
        "digit_count": sum(c.isdigit() for c in url),
        "subdomain_count": max(0, len(hostname.split(".")) - 2),
        "has_ip_address": has_ip,
        "uses_https": int(parsed.scheme == "https"),
        "suspicious_word_count": sum(
            word in url.lower() for word in suspicious_words
        ),
        "has_port": has_port
    }


HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>PhishGuard | URL Security</title>
<style>
* { box-sizing: border-box; }
body {
    margin: 0;
    font-family: Arial, sans-serif;
    background: #f1f5f9;
    color: #172033;
}
header {
    background: #101827;
    color: white;
    padding: 22px 8%;
    display: flex;
    justify-content: space-between;
    align-items: center;
}
.brand { font-size: 24px; font-weight: bold; }
.brand span { color: #4ade80; }
.status { color: #4ade80; font-size: 13px; }
main {
    max-width: 850px;
    margin: 55px auto;
    padding: 20px;
}
.hero { text-align: center; margin-bottom: 35px; }
.hero h1 { font-size: 34px; margin-bottom: 10px; }
.hero p { color: #64748b; }
.panel {
    background: white;
    padding: 32px;
    border-radius: 15px;
    box-shadow: 0 8px 25px rgba(0,0,0,.06);
}
label { display: block; font-weight: bold; margin-bottom: 12px; }
input {
    width: 100%;
    padding: 16px;
    border: 1px solid #cbd5e1;
    border-radius: 8px;
    font-size: 16px;
}
button {
    width: 100%;
    padding: 16px;
    margin-top: 15px;
    border: none;
    border-radius: 8px;
    background: #16a34a;
    color: white;
    font-size: 16px;
    font-weight: bold;
    cursor: pointer;
}
button:hover { background: #15803d; }
.result {
    margin-top: 25px;
    padding: 20px;
    border-radius: 10px;
    text-align: center;
}
.phishing { background: #fee2e2; color: #b91c1c; }
.legitimate { background: #dcfce7; color: #166534; }
.error { background: #fef3c7; color: #92400e; }
.result h2 { margin-top: 0; }
.info {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 15px;
    margin-top: 25px;
}
.info div {
    background: #f8fafc;
    padding: 18px;
    border-radius: 10px;
    text-align: center;
}
.info strong { display: block; margin-bottom: 7px; }
footer {
    text-align: center;
    color: #64748b;
    padding: 25px;
    font-size: 13px;
}
@media(max-width:600px) {
    .info { grid-template-columns: 1fr; }
    .hero h1 { font-size: 27px; }
}
</style>
</head>
<body>
<header>
    <div class="brand">🛡 Phish<span>Guard</span></div>
    <div class="status">● SECURITY ANALYSIS</div>
</header>

<main>
    <section class="hero">
        <h1>Phishing URL Detection</h1>
        <p>AI-powered protection against suspicious and fraudulent URLs.</p>
    </section>

    <section class="panel">
        <form method="POST">
            <label for="url">Enter a URL to analyse</label>
            <input
                type="text"
                id="url"
                name="url"
                placeholder="https://example.com"
                value="{{ url }}"
                required
            >
            <button type="submit">Analyse URL</button>
        </form>

        {% if result %}
        <div class="result {{ result_class }}">
            <h2>{{ result }}</h2>
            <p>{{ message }}</p>
            <strong>Prediction confidence: {{ confidence }}%</strong>
        </div>
        {% endif %}
    </section>

 <section class="info">

    <h2>How PhishGuard Works</h2>

    <p>PhishGuard analyses submitted URLs using machine learning.</p>

    <ol>

        <li>User enters a URL.</li>

        <li>The system extracts URL features.</li>

        <li>The Random Forest model analyses the features.</li>

        <li>The system displays the prediction.</li>

    </ol>

</section>.
       <section class="info">
        <div>
            <strong>Detection Model</strong>
            {{ model_name }}
        </div>
        <div>
            <strong>Analysis Method</strong>
            URL-based Machine Learning
        </div>
    </section>
</main>

<footer>
    PhishGuard | BSc Computer Science (Cybersecurity) Project
    <br>AI-Based Phishing URL Detection System
</footer>
</body>
</html>
"""


@app.route("/", methods=["GET", "POST"])
def home():
    result = ""
    result_class = ""
    message = ""
    confidence = ""
    url = ""

    if request.method == "POST":
        url = request.form.get("url", "").strip()

        if not url:
            result = "Invalid input"
            result_class = "error"
            message = "Please enter a URL."
        else:
            try:
                features = extract_url_features(url)
                input_data = pd.DataFrame([features]).reindex(
                    columns=feature_names
                )

                prediction = int(model.predict(input_data)[0])
                probabilities = model.predict_proba(input_data)[0]
                classes = list(model.classes_)
                predicted_index = classes.index(prediction)
                confidence = round(
                    float(probabilities[predicted_index]) * 100, 2
                )

                if prediction == 1:
                    result = "WARNING: PHISHING URL"
                    result_class = "phishing"
                    message = "This URL has been classified as potentially malicious."
                else:
                    result = "LEGITIMATE URL"
                    result_class = "legitimate"
                    message = "The model classified this URL as legitimate."

            except Exception:
                result = "Analysis error"
                result_class = "error"
                message = "Unable to analyse this URL. Check the input."

    return render_template_string(
        HTML,
        result=result,
        result_class=result_class,
        message=message,
        confidence=confidence,
        url=url,
       
analysis_details=features,
model_name=model_name 
    )


if __name__ == "__main__":
    app.run(debug=True)