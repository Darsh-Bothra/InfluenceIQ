# InfluenceIQ — End-to-End Project Documentation

**InfluenceIQ** is an AI-powered Instagram influencer ranking system. It measures and ranks influencers based on **engagement**, **structural credibility** (followers, verification, post volume), and **audience sentiment** (RoBERTa on comments), then surfaces results in a **Streamlit** dashboard.

The repository is intentionally small: two Jupyter notebooks (offline pipeline), one Streamlit app, CSV outputs, and raw `posts.json`. There is no backend API, database, or deployment configuration.

---

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Data Flow](#data-flow)
3. [File Reference](#file-reference)
4. [Stage 1: Dataset Processing (`dataset.ipynb`)](#stage-1-dataset-processing-datasetipynb)
5. [Stage 2: Ranking (`app.ipynb`)](#stage-2-ranking-appipynb)
6. [Stage 3: Dashboard (`app.py`)](#stage-3-dashboard-apppy)
7. [Data Files](#data-files)
8. [Scoring Model](#scoring-model)
9. [Technology Stack](#technology-stack)
10. [Running the Project](#running-the-project)
11. [Design Notes](#design-notes)

---

## Architecture Overview

```mermaid
flowchart TD
    A["Instagram scrape"]

    A --> B["dataset.json"]
    A --> C["posts.json"]

    B --> D["dataset.ipynb"]
    C --> D
    E["RoBERTa sentiment model"] --> D

    D --> F["instaInfluencer.csv"]
    F --> G["app.ipynb"]
    G --> H["AuraLevel.csv"]

    H --> I["app.py"]
    J["imagedb.csv"] --> I
    I --> K["End user"]
```

| Layer | Components |
|-------|------------|
| Raw data | `dataset.json` (not in repo), `posts.json` |
| Pipeline | `dataset.ipynb`, `app.ipynb`, RoBERTa model |
| Processed | `instaInfluencer.csv`, `AuraLevel.csv`, `imagedb.csv` |
| UI | `app.py` (Streamlit dashboard) |

---

## Data Flow

```mermaid
flowchart LR
    A["dataset.json\n+ posts.json"] --> B["dataset.ipynb"]
    B --> C["instaInfluencer.csv"]
    C --> D["app.ipynb"]
    D --> E["AuraLevel.csv\n+ imagedb.csv"]
    E --> F["app.py"]
```

---

## File Reference

| File | Role | Size / Records |
|------|------|----------------|
| `Readme.md` | Project overview: methodology, formulas, installation | 2.8 KB |
| `requirements.txt` | Streamlit app dependencies only | 5 packages |
| `dataset.ipynb` | Stage 1: ETL, engagement, credibility, sentiment | 751 KB |
| `app.ipynb` | Stage 2: normalization, ranking, export | 3 KB |
| `app.py` | Streamlit dashboard (reads precomputed CSVs) | 7.6 KB |
| `instaInfluencer.csv` | Intermediate processed dataset | ~978 rows |
| `AuraLevel.csv` | Final ranked output consumed by the UI | ~978 rows |
| `imagedb.csv` | Username → profile picture URL mapping | ~998 rows |
| `posts.json` | Scraped post data with comments | ~292K lines, 21 MB |
| `dataset.json` | Raw influencer scrape | **Not in repo** (referenced by notebook) |
| `assets/` | Placeholder directory for static assets | Empty |

### Project Structure

```
InfluenceIQ/
├── Readme.md              # High-level project README
├── requirements.txt       # Streamlit dependencies
├── dataset.ipynb          # ETL + engagement + credibility + sentiment
├── app.ipynb              # Normalization + ranking → AuraLevel.csv
├── app.py                 # Streamlit dashboard
├── instaInfluencer.csv    # Intermediate processed data
├── AuraLevel.csv          # Final ranked output (UI input)
├── imagedb.csv            # Profile picture URLs
├── posts.json             # Raw post/comment scrape
├── assets/                # (empty)
└── docs/
    └── architecture.md    # This document
```

---

## Stage 1: Dataset Processing (`dataset.ipynb`)

This notebook is the core ML and data pipeline. It expects `dataset.json` (Instagram profile scrape) and `posts.json` (post-level comments).

### Step 1 — Load and Clean Profile Data

```python
df = pd.read_json('dataset.json')
df = df[[
    "id", "inputUrl", "username", "businessCategoryName", "verified",
    "followersCount", "igtvVideoCount", "relatedProfiles",
    "latestIgtvVideos", "postsCount", "latestPosts"
]]
df = df.dropna(subset=['latestPosts'])
```

Rows without `latestPosts` are dropped. Heavy nested columns (`latestPosts`, `latestIgtvVideos`) are retained for feature extraction.

### Step 2 — Engagement Metrics

For each influencer (loops start at index 1, skipping row 0):

1. **avgLikes** — mean likes from `latestPosts`, plus IGTV likes
2. **avgComments** — mean comments from posts and IGTV
3. **avgViews** — mean `videoViewCount` from IGTV; if zero, fallback to `followersCount × 0.015`

**Engagement rate formula:**

```
engagementRate = (avgLikes + avgComments + avgViews) / followersCount
```

### Step 3 — Structural Credibility Score

```python
def calculate_credibility_score(verified, follower_count, num_posts, focusRow):
    threshold = df[focusRow].describe()["50%"]  # median postsCount
    verified_bonus = 50 if verified == 1 else 0
    follower_score = 10 * math.log10(follower_count + 1)
    posts_score = 5 * math.log10(num_posts + 1)
    post_multiplier = min(1, num_posts / threshold)
    adjusted_follower_score = follower_score * post_multiplier
    return verified_bonus + adjusted_follower_score + posts_score
```

| Component | Description |
|-----------|-------------|
| Verified bonus | +50 for verified accounts |
| Follower score | `10 × log₁₀(followers + 1)`, scaled by post activity |
| Post multiplier | `min(1, postsCount / median_posts)` — dampens follower score for inactive accounts |
| Posts score | `5 × log₁₀(posts + 1)` |

### Step 4 — Comment Extraction from `posts.json`

```python
post = pd.read_json('posts.json')
post = post.dropna(subset=['latestComments'])

for i in range(1, len(post)):
    inputUrl = post.iloc[i]["inputUrl"]
    L = [comment['text'] for comment in post.iloc[i]['latestComments']]
    # Match by inputUrl and append comments to df.latestComments
```

`posts.json` contains both error records (`"error": "not_found"`) and valid posts. Only rows with `latestComments` are used. Comments are aggregated per influencer into a `latestComments` list of strings.

### Step 5 — Sentiment-Based `credScore` (RoBERTa)

**Model:** `delarosajav95/tw-roberta-base-sentiment-FT-v2` via Hugging Face `transformers.pipeline`

For each influencer's comments:

1. Truncate long text to the model's maximum token length
2. Classify each comment as Positive, Neutral, or Negative (highest-confidence label wins)
3. Count dominant sentiments per category
4. Compute scores:

```
net_sentiment = (Positive - Negative) / (Positive + Negative)

engagement    = (Positive + Negative) / total_comments

credScore     = 50 + 50 × (0.7 × net_sentiment + 0.3 × engagement)
```

The score is clamped to `[0, 100]`. Influencers with no comments receive a default score of **50**.

### Step 6 — Export

Nested columns (`latestPosts`, `latestIgtvVideos`, `relatedProfiles`) are dropped and the result is written to **`instaInfluencer.csv`**.

**Columns in `instaInfluencer.csv`:**

`id`, `inputUrl`, `username`, `businessCategoryName`, `verified`, `followersCount`, `igtvVideoCount`, `postsCount`, `avgLikes`, `avgComments`, `avgViews`, `engagementRate`, `credibilityScore`, `latestComments`, `credScore`

---

## Stage 2: Ranking (`app.ipynb`)

Reads `instaInfluencer.csv` and produces the final leaderboard.

### Blended Credibility

```python
df["credScore"] = 0.6 * df["credibilityScore"] + 0.4 * df["credScore"]
```

Structural credibility (60%) is combined with sentiment credibility (40%).

### Min–Max Normalization

```python
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
df['norm_engagement'] = scaler.fit_transform(df[['engagementRate']])
df['norm_cred'] = scaler.fit_transform(df[['credScore']])
```

Both metrics are scaled to the range `[0, 1]`.

### Weighted Ranking Score

```python
weight_engagement = 0.15
weight_cred = 0.85

df['ranking_score'] = weight_engagement * df['norm_engagement'] + weight_cred * df['norm_cred']
df['rank'] = df['ranking_score'].rank(method='dense', ascending=False)
```

Credibility dominates the final score (85% weight). Dense ranking ensures tied influencers share the same rank.

### Export

Writes **`AuraLevel.csv`** with additional columns: `norm_engagement`, `norm_cred`, `ranking_score`, `rank`.

**Sample top ranks:** `instagram` (#1), `kyliejenner` (#2), `cristiano` (#3).

---

## Stage 3: Dashboard (`app.py`)

A read-only Streamlit UI over precomputed CSVs. No live scoring or model inference occurs at runtime.

### Data Loading

```python
data = pd.read_csv("AuraLevel.csv")
imagedb = pd.read_csv("imagedb.csv")[["username", "profilePicUrlHD"]]
sorted_data = data.sort_values(by="rank", ascending=True)
```

### Features

| Feature | Implementation |
|---------|----------------|
| Category filter | Sidebar `selectbox` on `businessCategoryName` |
| Pagination | 10 influencers per page |
| Leaderboard table | HTML table with clickable Instagram profile links |
| Search | Username substring match opens a detail view |
| Influencer detail | Profile photo (fetched from CDN), metrics, bar chart |
| Comparison charts | Grouped bar chart (engagement vs. credibility); scatter plot by category |

### Key Functions

**`show_influencer_details(influencer, imagedb)`**

- Fetches profile image from `imagedb.csv` via `requests` and `PIL`
- Displays Aura rank, followers, likes, comments, engagement rate, and credibility
- Renders a Plotly bar chart of raw metrics

**`format_number(n)`**

Formats large numbers for display: `686569041` → `686.6M`, `15000` → `15.0K`.

### Sidebar Controls

- **Filter by Category** — narrows the leaderboard to a business category
- **Page number** — paginates results (10 per page)
- **Search Influencer** — finds a specific user by name; shows detail panel on match

---

## Data Files

### `AuraLevel.csv` (~978 influencers)

Final ranked dataset consumed by the dashboard.

| Column | Description |
|--------|-------------|
| `rank` | Aura rank (1 = highest) |
| `engagementRate` | Raw engagement ratio |
| `credibilityScore` | Structural credibility |
| `credScore` | Sentiment-adjusted credibility (blended in `app.ipynb`) |
| `norm_engagement` | Normalized engagement `[0, 1]` |
| `norm_cred` | Normalized credibility `[0, 1]` |
| `ranking_score` | Weighted composite score |

### `imagedb.csv` (~998 rows)

Maps `username` → `profilePicUrlHD` (Instagram CDN URLs). Used exclusively for profile images in the UI.

### `posts.json` (~292K lines)

Array of post objects from scraping. Two record types:

- **Error records:** `{ "username", "error": "not_found", "errorDescription": "..." }`
- **Valid posts:** `inputUrl`, `latestComments` (array of `{ "text": "..." }`), plus metadata (`likesCount`, `videoViewCount`, etc.)

### `dataset.json` (not in repository)

Expected raw influencer scrape with nested `latestPosts` and `latestIgtvVideos`. Likely omitted due to file size. Required for a full re-run of `dataset.ipynb`.

---

## Scoring Model

```mermaid
flowchart TD
    A["Likes + Comments + Views"] --> B["engagementRate"]
    B --> C["norm_engagement"]

    D["Verified + Followers + Posts"] --> E["credibilityScore"]
    F["Audience comments"] --> G["RoBERTa classification"]
    G --> H["credScore"]

    E --> I["Blend scores\n0.6 × struct + 0.4 × sentiment"]
    H --> I
    I --> J["norm_cred"]

    C --> K["ranking_score\n0.15 × engagement + 0.85 × cred"]
    J --> K
    K --> L["Aura rank"]
```

Blend: `0.6 x credibilityScore + 0.4 x credScore`  
Final: `0.15 x norm_engagement + 0.85 x norm_cred`

### Formula Summary

| Metric | Formula |
|--------|---------|
| Engagement rate | `(avgLikes + avgComments + avgViews) / followersCount` |
| Structural credibility | `verified_bonus + adjusted_follower_score + posts_score` |
| Sentiment credibility | `50 + 50 × (0.7 × net_sentiment + 0.3 × engagement)` |
| Blended credibility | `0.6 × credibilityScore + 0.4 × credScore` |
| Ranking score | `0.15 × norm_engagement + 0.85 × norm_cred` |

---

## Technology Stack

| Layer | Technology |
|-------|------------|
| Data processing | Pandas, NumPy |
| ML / NLP | Hugging Face Transformers, PyTorch, RoBERTa |
| Normalization / ranking | scikit-learn `MinMaxScaler` |
| Visualization (app) | Plotly Express |
| Web UI | Streamlit |
| Image loading | Pillow, `requests` |

---

## Running the Project

### Prerequisites

**For the Streamlit dashboard:**

```bash
pip install -r requirements.txt
streamlit run app.py
```

**For the full pipeline (notebooks):**

```bash
pip install pandas numpy scikit-learn transformers torch ipykernel
```

### Typical Workflow

```
1. Scrape Instagram data  →  dataset.json + posts.json
2. Run dataset.ipynb      →  instaInfluencer.csv
3. Run app.ipynb          →  AuraLevel.csv
4. streamlit run app.py   →  interactive leaderboard
```

---

## Design Notes

1. **Batch offline pipeline** — Notebooks precompute all scores; `app.py` only reads CSVs at runtime.
2. **Credibility-heavy ranking** — 85% weight on credibility vs. 15% on engagement favors established, well-perceived accounts over viral one-hit profiles.
3. **Dual credibility signals** — Structural (`credibilityScore`) and sentiment (`credScore`) are blended before ranking.
4. **Missing `dataset.json`** — Re-running `dataset.ipynb` from scratch requires this file or an equivalent scrape.
5. **Empty `assets/` directory** — No static images or branding assets are checked in.
6. **Live-fetched profile images** — Instagram CDN URLs in `imagedb.csv` may expire over time.
7. **Index-1 loop quirk** — Several loops in `dataset.ipynb` start at index 1, skipping the first row. This may be intentional or a data artifact.

---

## Conclusion

InfluenceIQ ingests scraped Instagram data, computes engagement and dual credibility scores (structural + sentiment via RoBERTa), ranks influencers with a weighted formula in Jupyter notebooks, and presents the results in a Streamlit leaderboard with filters, search, pagination, and interactive charts.

The system is designed to help brands, investors, and analysts identify influencers with **sustained, credible impact** rather than relying solely on follower counts or raw likes.
