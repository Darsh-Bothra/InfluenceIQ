# 🚀 InfluenceIQ - AI-Powered Influencer Ranking

InfluenceIQ is an AI-powered influencer ranking system that evaluates
influencers based on **engagement**, **credibility**, and **audience
sentiment** instead of relying solely on traditional popularity metrics
such as followers and likes.

Many influencers achieve temporary viral success but fail to maintain
long-term audience trust. InfluenceIQ combines engagement metrics,
credibility indicators, and AI-driven sentiment analysis to identify
influencers with genuine and sustainable influence.

------------------------------------------------------------------------

## ✨ Features

-   📊 **Data Processing & Cleaning**
    -   Handles missing values
    -   Performs feature engineering
    -   Normalizes numerical features
-   ❤️ **Engagement Scoring**
    -   Calculates influencer engagement using average likes, comments,
        and views.
-   🏆 **Credibility Scoring**
    -   Uses follower count, number of posts, verification status, and
        audience sentiment to calculate a credibility score.
-   🤖 **AI-Powered Sentiment Analysis**
    -   Uses the **tw-roberta-base-sentiment-FT-v2** model to classify
        audience comments as:
        -   Positive
        -   Neutral
        -   Negative
-   📈 **AI-Based Influencer Ranking**
    -   Combines engagement and credibility using a weighted scoring
        algorithm to generate the final influencer rankings.

------------------------------------------------------------------------

# ⚙️ Methodology

## 1. Dataset Processing (`dataset.ipynb`)

The preprocessing pipeline performs the following tasks:

-   Loads and cleans influencer data.
-   Handles missing values.
-   Computes:
    -   Average Likes
    -   Average Comments
    -   Average Views

**Engagement Rate**

``` text
Engagement Rate = (Average Likes + Average Comments + Average Views) / Followers
```

**Credibility Score**

``` text
Credibility = Verification Bonus + 10 × log10(Followers + 1) + 5 × log10(Posts + 1)
```

------------------------------------------------------------------------

## 2. Influencer Ranking (`app.ipynb`)

Features are normalized using Min-Max Normalization.

``` text
X' = (X - Xmin) / (Xmax - Xmin)
```

Final Ranking Score:

``` text
Ranking Score = 0.15 × Normalized Engagement + 0.85 × Normalized Credibility
```

Influencers are ranked using the **Dense Ranking** method.

------------------------------------------------------------------------

## 3. Sentiment Analysis

Audience comments are analyzed using the
**tw-roberta-base-sentiment-FT-v2** model.

Each comment is classified as:

-   Positive
-   Neutral
-   Negative

Net Sentiment:

``` text
Net Sentiment = (Positive − Negative) / (Positive + Negative)
```

The sentiment score is incorporated into the credibility score to
improve ranking accuracy.

------------------------------------------------------------------------

# 🛠️ Tech Stack

-   Python
-   Pandas
-   NumPy
-   Scikit-learn
-   PyTorch
-   Hugging Face Transformers
-   RoBERTa (`tw-roberta-base-sentiment-FT-v2`)
-   Jupyter Notebook
-   uv

------------------------------------------------------------------------

# 📂 Project Structure

``` text
InfluenceIQ/
│
├── app.ipynb
├── dataset.ipynb
├── pyproject.toml
├── uv.lock
├── README.md
└── data/
```

------------------------------------------------------------------------

# 📦 Installation

## Clone the repository

``` bash
git clone https://github.com/Darsh-Bothra/InfluenceIQ.git
cd InfluenceIQ
```

## Install dependencies

``` bash
uv sync
```

If you don't have **uv** installed:

``` bash
pip install uv
```

------------------------------------------------------------------------

# ▶️ Running the Project

Activate the virtual environment.

**Linux / macOS**

``` bash
source .venv/bin/activate
```

**Windows**

``` powershell
.venv\Scripts\activate
```

Launch Jupyter Notebook:

``` bash
jupyter notebook
```

Run:

1.  `dataset.ipynb`
2.  `app.ipynb`

------------------------------------------------------------------------

# 🎯 Applications

-   Influencer Marketing
-   Brand Collaboration
-   Social Media Analytics
-   Creator Discovery
-   Audience Trust Analysis

------------------------------------------------------------------------

# 🔮 Future Improvements

-   Multi-platform support
-   Fake follower detection
-   Real-time ranking
-   Explainable AI
-   Streamlit dashboard
-   Trend prediction

------------------------------------------------------------------------

# 🤝 Contributing

Contributions are welcome! Feel free to fork the repository, make
improvements, and submit a Pull Request.

------------------------------------------------------------------------

# 📜 License

This project is intended for educational and research purposes.

------------------------------------------------------------------------

## ⭐ If you found this project helpful, consider giving it a star!
