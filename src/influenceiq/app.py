import streamlit as st
import pandas as pd
import plotly.express as px
import math
import requests
from PIL import Image
from io import BytesIO

from paths import DATA_PROCESSED

headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0 Safari/537.36"
}

data = pd.read_csv(DATA_PROCESSED / "AuraLevel.csv")
imagedb = pd.read_csv(DATA_PROCESSED / "imagedb.csv")
imagedb = imagedb[["username", "profilePicUrlHD"]]
# Sort by Rank
sorted_data = data.sort_values(by="rank", ascending=True)

# Sidebar Filter for Category
dropdown_options = sorted_data["businessCategoryName"].dropna().unique()
st.sidebar.header("Filter by Category")
selected_category = st.sidebar.selectbox("Select Category", options=["All"] + list(dropdown_options))
if selected_category != "All":
    sorted_data = sorted_data[sorted_data["businessCategoryName"].str.contains(selected_category, na=False)]

# Pagination Controls: display data in intervals of 10
page_size = 10
total_pages = math.ceil(len(sorted_data) / page_size)
page_number = st.sidebar.number_input("Page number", min_value=1, max_value=total_pages, value=1, step=1)
start_idx = (page_number - 1) * page_size
end_idx = start_idx + page_size
paginated_data = sorted_data.iloc[start_idx:end_idx].copy()

def show_influencer_details(influencer, imagedb):
    st.markdown("---")
    
    # Layout for profile image and username with verified blue tick
    col_image, col_info = st.columns([1, 4])
    with col_image:
        # Display profile photo if available, else a placeholder image
        profile_photo_url = imagedb[imagedb["username"] == influencer["username"]]   ["profilePicUrlHD"].values[0]
        if profile_photo_url and isinstance(profile_photo_url, str) and profile_photo_url.strip():
            response = requests.get(profile_photo_url, headers=headers)
            response.raise_for_status()  # Raises an error for bad responses
            image = Image.open(BytesIO(response.content))
            st.image(image, width=100)
        
    
    with col_info:
        verified_icon_url = "https://5.imimg.com/data5/ANDROID/Default/2021/3/SN/FW/NM/68340479/product-jpeg.jpeg"
            
        verified_icon = f" <img src='{verified_icon_url}' style='width:25px;height:25px; vertical-align:middle;border-radius:50%;'>"
        # Create a custom header combining username and verified icon
        username_html = f"<h2 style='margin-bottom:0;'>{influencer['username']}{verified_icon}</h2>"
        st.markdown(username_html, unsafe_allow_html=True)
        # Display clickable Instagram link below the username
        st.markdown(f"[Visit Instagram Profile]({influencer['inputUrl']})", unsafe_allow_html=True)

    # Two-column layout for influencer metrics
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"**Aura:** {influencer['rank']}")
        st.markdown(f"**Followers:** {format_number(influencer['followersCount'])}")
        st.markdown(f"**Avg Likes:** {format_number(influencer['avgLikes'])}")
    with col2:
        st.markdown(f"**Avg Comments:** {format_number(influencer['avgComments'])}")
        st.markdown(f"**Engagement Rate:** {influencer['engagementRate']:.2f}")
        st.markdown(f"**Credibility Score:** {influencer['norm_cred']:.2f}")

    # Bar chart for influencer's key metrics using raw values from the data
    metrics = {
        "Metric": ["Followers", "Avg Likes", "Avg Comments", "Engagement Rate", "Credibility Score"],
        "Value": [
            influencer["followersCount"],
            influencer["avgLikes"],
            influencer["avgComments"],
            influencer["engagementRate"],
            influencer["norm_cred"],
        ],
    }
    df_metrics = pd.DataFrame(metrics)
    fig_detail = px.bar(
        df_metrics,
        x="Metric",
        y="Value",
        title=f"{influencer['username']}'s Metrics",
        color="Metric",
        text_auto=True
    )
    st.plotly_chart(fig_detail)
    st.markdown("---")

# Function to format numbers into K, M, or B
def format_number(n):
    try:
        n = float(n)
        if n >= 1e9:
            return f"{n/1e9:.1f}B"
        elif n >= 1e6:
            return f"{n/1e6:.1f}M"
        elif n >= 1e3:
            return f"{n/1e3:.1f}K"
        else:
            return f"{n:.0f}"
    except (ValueError, TypeError):
        return n

# Format numeric columns for display in the main leaderboard
for col in ["followersCount", "avgLikes", "avgComments"]:
    paginated_data[col] = paginated_data[col].apply(format_number)
for col in ["engagementRate", "credibilityScore", "norm_cred"]:
    paginated_data[col] = paginated_data[col].apply(lambda x: f"{x:.2f}")
# Create a new column with clickable Instagram link for display in leaderboard
paginated_data["Profile"] = paginated_data.apply(
    lambda row: f'<a href="{row["inputUrl"]}" target="_blank">{row["username"]}</a>', axis=1
)
display_cols = ["rank", "Profile", "followersCount", "avgLikes", "avgComments", "engagementRate", "credibilityScore"]
paginated_display = paginated_data[display_cols].rename(columns={
    "rank": "Aura",
    "Profile": "Influencer",
    "avgLikes": "Likes",
    "followersCount": "Followers",
    "avgComments": "Comments",
    "engagementRate": "Engagement Rate",
    "credibilityScore": "Credibility Score"
})
# ---------------------
# Dedicated Influencer Page
# ---------------------

# --- Dedicated Influencer Detail Page ---
st.sidebar.subheader("🔍 Search Influencer")
search_query = st.sidebar.text_input("Enter Influencer Name")

if search_query:
    search_results = sorted_data[sorted_data["username"].str.contains(search_query, case=False, na=False)].copy()
    if not search_results.empty:
        # If a single match is found, display its details directly
        if len(search_results) == 1:
            influencer_detail = search_results.iloc[0]
            with st.expander("View Influencer Details", expanded=True):
                show_influencer_details(influencer_detail, imagedb)
        else:
            # Multiple matches: let the user select one
            selected_influencer = st.sidebar.selectbox("Select Influencer", options=search_results["username"].tolist())
            influencer_detail = search_results[search_results["username"] == selected_influencer].iloc[0]
            with st.expander("View Influencer Details", expanded=True):
                show_influencer_details(influencer_detail, imagedb)
    else:
        st.sidebar.write("No influencer found with that name.")
# Title and Leaderboard
st.title("📢 Top Influencers Ranking")
st.write("Influencers ranked based on engagement and credibility.")
st.subheader("🏆 Influencer Leaderboard")
st.write(f"Showing page {page_number} of {total_pages}")
st.markdown(paginated_display.to_html(escape=False, index=False), unsafe_allow_html=True)

# Bar Chart: Engagement & Credibility Comparison for all influencers
st.subheader("📊 Engagement & Credibility Comparison")
fig_bar = px.bar(sorted_data, x="username", y=["engagementRate", "credibilityScore"],
                 title="Influencer Engagement vs. Credibility",
                 barmode="group", color_discrete_map={"engagementRate": "blue", "credibilityScore": "red"})
st.plotly_chart(fig_bar)

# Scatter Plot: Engagement vs. Credibility for all influencers
st.subheader("📊 Engagement vs. Credibility Scatter Plot")
fig_scatter = px.scatter(sorted_data, x="engagementRate", y="credibilityScore",
                         size="followersCount", color="businessCategoryName",
                         hover_data=["username", "rank"],
                         title="Engagement vs. Credibility")
st.plotly_chart(fig_scatter)


