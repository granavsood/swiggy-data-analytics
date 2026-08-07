import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import joblib

# Must be the first Streamlit command
st.set_page_config(page_title="Swiggy Analytics Dashboard", page_icon="🍔", layout="wide", initial_sidebar_state="collapsed")

st.markdown("""
<style>
    .kpi-card {
        background-color: #1E1E1E !important;
        border-radius: 8px;
        padding: 20px 10px;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
        border-top: 5px solid #ED7D31;
        margin-bottom: 20px;
    }
    .kpi-value {
        font-size: 34px;
        font-weight: 900;
        color: #ED7D31 !important;
        line-height: 1.2;
    }
    .kpi-label {
        font-size: 14px;
        color: #CCCCCC !important;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        font-weight: 700;
        margin-top: 5px;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
        margin-bottom: 20px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: transparent;
        border-radius: 4px 4px 0px 0px;
        gap: 1px;
        padding-top: 10px;
        padding-bottom: 10px;
        font-weight: 600;
        color: #CCCCCC;
    }
    .stTabs [aria-selected="true"] {
        background-color: #ED7D31 !important;
        color: white !important;
        border-radius: 4px;
        padding: 0 15px;
    }
</style>
""", unsafe_allow_html=True)

# Un-reversed color arrays (Highest value gets the Darkest Orange)
pbi_continuous_oranges = ['#FCE4D6', '#F8CBAD', '#F4B183', '#ED7D31', '#C55A11', '#9E480E']
pbi_discrete_oranges = ['#ED7D31', '#F4B183', '#FCE4D6', '#C55A11', '#F8CBAD']

def get_hardcoded_colors(series):
    vals = series.tolist()
    min_val = min(vals)
    max_val = max(vals)
    if min_val == max_val: 
        return [pbi_continuous_oranges[-1]] * len(vals)
    
    colors = []
    for v in vals:
        norm = (v - min_val) / (max_val - min_val)
        idx = int(norm * 5)
        idx = max(0, min(idx, 5))
        colors.append(pbi_continuous_oranges[idx])
    return colors

def apply_powerbi_style(fig, title="", xtitle=None, ytitle=None):
    fig.update_layout(
        title=dict(text=f"<b>{title}</b>", font=dict(size=18, color="#ED7D31", family="sans-serif"), x=0.5, y=0.95),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family="sans-serif", size=13, color="#CCCCCC"),
        margin=dict(r=50, t=50),
        xaxis=dict(showgrid=False, zeroline=False, title=xtitle if xtitle else "", showticklabels=True, automargin=True),
        yaxis=dict(showgrid=False, zeroline=False, title=ytitle if ytitle else "", showticklabels=True, automargin=True),
        coloraxis_showscale=False
    ) 
    fig.update_traces(selector=dict(type="bar"), cliponaxis=False)
    fig.update_traces(selector=dict(type="scatter"), cliponaxis=False)
    return fig

@st.cache_data
def load_data():
    orders = pd.read_csv('Processed_Data/processed_orders.csv')
    orders['order_date'] = pd.to_datetime(orders['order_date'])
    users_rfm = pd.read_csv('Processed_Data/processed_users_rfm.csv')
    menu = pd.read_csv('Data/Menu.csv')
    food = pd.read_csv('Data/Food.csv')
    menu_food = pd.merge(menu, food, on='Food_id', how='left')
    return orders, users_rfm, menu_food

try:
    df_orders, df_users, df_menu = load_data()
except Exception as e:
    st.error(f"Error loading data: {e}")
    st.stop()

col1, col2 = st.columns([1, 15])
with col1:
    st.markdown("<h1 style='color: #ED7D31; font-size: 45px; margin-top:-10px;'>🍔</h1>", unsafe_allow_html=True)
with col2:
    st.markdown("<h1 style='color: #ED7D31; margin:0; padding:0; line-height:1; font-weight:900;'>SWIGGY</h1>", unsafe_allow_html=True)
    st.markdown("<h4 style='color: #AAAAAA; margin:0; padding:0; font-weight:600;'>Swiggy Karo, Phir Jo Chahe Karo!</h4>", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "1. Overview", "2. User Performance", "3. City Overview", 
    "4. Restaurant Analysis", "5. Insights", "6. ML Predictor"
])

with tab1:
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    total_orders = len(df_orders)
    total_users = df_orders['User_id'].nunique()
    total_sales = df_orders['Sales_amount'].sum()
    total_rating_count = df_orders['Rating'].count()
    
    kpi1.markdown(f"<div class='kpi-card'><div class='kpi-value'>{total_orders/1000:.0f}K</div><div class='kpi-label'>Orders Count</div></div>", unsafe_allow_html=True)
    kpi2.markdown(f"<div class='kpi-card'><div class='kpi-value'>{total_users/1000:.0f}K</div><div class='kpi-label'>User Count</div></div>", unsafe_allow_html=True)
    kpi3.markdown(f"<div class='kpi-card'><div class='kpi-value'>{total_sales/1e6:.0f}M</div><div class='kpi-label'>Current YR Sales</div></div>", unsafe_allow_html=True)
    kpi4.markdown(f"<div class='kpi-card'><div class='kpi-value'>{total_rating_count/1000:.0f}K</div><div class='kpi-label'>Rating Count</div></div>", unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 1])
    with col1:
        food_stats = df_menu.groupby('Food_Type').agg(Avg_Price=('Price', 'mean')).reset_index().sort_values(by='Avg_Price')
        fig_food = go.Figure(go.Bar(
            x=food_stats['Food_Type'], y=food_stats['Avg_Price'],
            text=food_stats['Avg_Price'].apply(lambda x: f"₹{x:.1f}"), textposition='outside', width=0.5,
            marker=dict(color=get_hardcoded_colors(food_stats['Avg_Price']))
        ))
        fig_food = apply_powerbi_style(fig_food, "Average Price by Food Type", xtitle="Food Type", ytitle="Average Price (₹)")
        st.plotly_chart(fig_food, use_container_width=True, theme=None)
        
    with col2:
        yearly_sales = df_orders.groupby(df_orders['order_date'].dt.year)['Sales_amount'].sum().reset_index()
        fig_year = go.Figure(go.Scatter(
            x=yearly_sales['order_date'], y=yearly_sales['Sales_amount'],
            mode='lines+markers', line=dict(width=3, color='#ED7D31'), marker=dict(size=10, color='#9E480E')
        ))
        fig_year = apply_powerbi_style(fig_year, "Amount By Year", xtitle="Year", ytitle="Sales Amount (₹)")
        fig_year.update_xaxes(dtick=1)
        st.plotly_chart(fig_year, use_container_width=True, theme=None)

    top_cities = df_orders.groupby('City')['Sales_amount'].sum().nlargest(10).reset_index().sort_values(by='Sales_amount', ascending=True)
    fig_cities = go.Figure(go.Bar(
        x=top_cities['Sales_amount'], y=top_cities['City'], orientation='h',
        text=top_cities['Sales_amount'].apply(lambda x: f"{x/1e6:.0f}M"), textposition='outside',
        marker=dict(color=get_hardcoded_colors(top_cities['Sales_amount']))
    ))
    fig_cities = apply_powerbi_style(fig_cities, "Top 10 City Amount")
    st.plotly_chart(fig_cities, use_container_width=True, theme=None)

with tab2:
    col1, col2 = st.columns(2)
    with col1:
        sales_marital = df_orders.groupby('Marital Status')['Sales_amount'].sum().reset_index().sort_values(by='Sales_amount')
        fig_marital = go.Figure(go.Bar(
            x=sales_marital['Marital Status'], y=sales_marital['Sales_amount'],
            text=sales_marital['Sales_amount'].apply(lambda x: f"{x/1e6:.0f}M"), textposition='outside', width=0.5,
            marker=dict(color=get_hardcoded_colors(sales_marital['Sales_amount']))
        ))
        fig_marital = apply_powerbi_style(fig_marital, "Sales By Marital Status", xtitle="Marital Status", ytitle="Sales Amount")
        st.plotly_chart(fig_marital, use_container_width=True, theme=None)
        
    with col2:
        sales_occ = df_orders.groupby('Occupation')['Sales_amount'].sum().reset_index().sort_values(by='Sales_amount', ascending=False)
        fig_occ = go.Figure(go.Pie(
            labels=sales_occ['Occupation'], values=sales_occ['Sales_amount'], hole=0.6,
            marker=dict(colors=pbi_discrete_oranges), textinfo='percent+label', textposition='outside'
        ))
        fig_occ = apply_powerbi_style(fig_occ, "Sale By Occupation")
        fig_occ.update_layout(showlegend=False)
        st.plotly_chart(fig_occ, use_container_width=True, theme=None)
        
    users_age = df_orders.groupby('Age')['User_id'].nunique().reset_index()
    fig_age = go.Figure(go.Bar(
        x=users_age['Age'], y=users_age['User_id'],
        text=users_age['User_id'].apply(lambda x: f"{x/1000:.1f}K" if x>1000 else f"{x}"), textposition='outside',
        marker=dict(color=get_hardcoded_colors(users_age['User_id']))
    ))
    fig_age = apply_powerbi_style(fig_age, "Users By Age", xtitle="Age", ytitle="User Count")
    fig_age.update_xaxes(type='category') # Ensure X axis is categorical so bars align perfectly
    st.plotly_chart(fig_age, use_container_width=True, theme=None)

with tab3:
    all_cities_raw = sorted(df_orders['City'].dropna().unique())
    selected_city = st.selectbox("Select City", ["All"] + all_cities_raw)
    
    if selected_city != "All":
        city_df = df_orders[df_orders['City'] == selected_city]
    else:
        city_df = df_orders

    tot_cities = city_df['City'].nunique()
    users_sales = city_df.groupby('User_id')['Sales_amount'].sum().sort_values(ascending=False)
    top_10_count = max(1, int(len(users_sales) * 0.1))
    top10_sales = users_sales.head(top_10_count).sum()
    
    max_yr = city_df['order_date'].dt.year.max()
    curr_yr_sales = city_df[city_df['order_date'].dt.year == max_yr]['Sales_amount'].sum()
    prev_yr_sales = city_df[city_df['order_date'].dt.year == (max_yr - 1)]['Sales_amount'].sum()
    
    ckpi1, ckpi2, ckpi3, ckpi4 = st.columns(4)
    ckpi1.markdown(f"<div class='kpi-card'><div class='kpi-value'>{tot_cities}</div><div class='kpi-label'>Total City</div></div>", unsafe_allow_html=True)
    ckpi2.markdown(f"<div class='kpi-card'><div class='kpi-value'>{top10_sales/1e6:.0f}M</div><div class='kpi-label'>Top 10% Customer</div></div>", unsafe_allow_html=True)
    ckpi3.markdown(f"<div class='kpi-card'><div class='kpi-value'>{curr_yr_sales/1e6:.0f}M</div><div class='kpi-label'>Current YR Sales</div></div>", unsafe_allow_html=True)
    ckpi4.markdown(f"<div class='kpi-card'><div class='kpi-value'>{prev_yr_sales/1e6:.0f}M</div><div class='kpi-label'>Previous YR Sales</div></div>", unsafe_allow_html=True)

    city_stats_filtered = city_df.groupby('City').agg(Sale=('Sales_amount', 'sum'), Orders=('User_id', 'count')).reset_index()
    city_yearly = city_df.groupby(['City', city_df['order_date'].dt.year])['Sales_amount'].sum().unstack(fill_value=0)
    city_stats_filtered['Current_YR_Sales'] = city_stats_filtered['City'].map(city_yearly[max_yr] if max_yr in city_yearly.columns else pd.Series(0, index=city_stats_filtered['City'])).fillna(0)
    city_stats_filtered['Previous_YR_Sale'] = city_stats_filtered['City'].map(city_yearly[max_yr - 1] if (max_yr - 1) in city_yearly.columns else pd.Series(0, index=city_stats_filtered['City'])).fillna(0)
    
    map_col, table_col = st.columns([1, 1])
    with map_col:
        city_coords = {
            'Delhi': (28.7041, 77.1025), 'Bangalore': (12.9716, 77.5946), 'Ahmedabad': (23.0225, 72.5714),
            'Pune': (18.5204, 73.8567), 'Tirupati': (13.6288, 79.4192), 'Chennai': (13.0827, 80.2707),
            'Hyderabad': (17.3850, 78.4867), 'Raipur': (21.2514, 81.6296), 'Surat': (21.1702, 72.8311),
            'Sultanpur': (26.2561, 82.0722), 'Gurgaon': (28.4595, 77.0266), 'Bikaner': (28.0229, 73.3119),
            'Tirupur': (11.1085, 77.3411), 'Sirsa': (29.5330, 75.0177), 'Sonipat': (28.9931, 77.0151),
            'Vizag': (17.6868, 83.2185), 'Ranchi': (23.3441, 85.3096), 'Vijayawada': (16.5062, 80.6480),
            'Agra': (27.1767, 78.0081), 'Rajahmundry': (17.0005, 81.8040), 'Mumbai': (19.0760, 72.8777),
            'Kolkata': (22.5726, 88.3639), 'Jaipur': (26.9124, 75.7873), 'Lucknow': (26.8467, 80.9462),
            'Kanpur': (26.4499, 80.3319), 'Nagpur': (21.1458, 79.0882), 'Indore': (22.7196, 75.8577),
            'Thane': (19.2183, 72.9781), 'Bhopal': (23.2599, 77.4126), 'Patna': (25.5941, 85.1376)
        }
        import plotly.express as px
        map_data = []
        for c, row in city_stats_filtered.iterrows():
            clean_city = row['City'].split(',')[-1].strip()
            if clean_city in city_coords:
                map_data.append({'City': row['City'], 'lat': city_coords[clean_city][0], 'lon': city_coords[clean_city][1], 'Sales': row['Sale']})
        if map_data:
            map_df = pd.DataFrame(map_data)
            fig_map = px.scatter_mapbox(
                map_df, lat="lat", lon="lon", size="Sales", color="Sales",
                hover_name="City", hover_data={"lat":False, "lon":False, "Sales":True},
                color_continuous_scale=pbi_continuous_oranges,
                size_max=25, zoom=3.5, center={"lat": 22.0, "lon": 79.0}
            )
            fig_map.update_layout(mapbox_style="carto-darkmatter", margin=dict(r=0, t=0, l=0, b=0), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", coloraxis_showscale=False)
            st.plotly_chart(fig_map, use_container_width=True, theme=None, height=350)
        else:
            st.info("Map data not available for this selection.")
    
    with table_col:
        st.dataframe(city_stats_filtered, use_container_width=True, hide_index=True, height=350)
        
    st.markdown("<br>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    
    city_stats_bars = city_df.groupby('City').agg(
        Sales=('Sales_amount', 'sum'), Orders=('User_id', 'count'), Rating_Count=('Rating', 'count')
    ).reset_index().sort_values(by='Sales', ascending=False)
    
    with col1:
        top10_sales = city_stats_bars.head(10).sort_values(by='Sales', ascending=True)
        fig_c1 = go.Figure(go.Bar(
            x=top10_sales['Sales'], y=top10_sales['City'], orientation='h',
            text=top10_sales['Sales'].apply(lambda x: f"{x/1e6:.0f}M"), textposition='outside',
            marker=dict(color=get_hardcoded_colors(top10_sales['Sales']))
        ))
        fig_c1 = apply_powerbi_style(fig_c1, "Sales by City")
        st.plotly_chart(fig_c1, use_container_width=True, theme=None)
    with col2:
        users_city = city_df.groupby('City')['User_id'].nunique().nlargest(10).reset_index().sort_values(by='User_id', ascending=True)
        fig_c2 = go.Figure(go.Bar(
            x=users_city['User_id'], y=users_city['City'], orientation='h',
            text=users_city['User_id'].apply(lambda x: f"{x/1000:.1f}K"), textposition='outside',
            marker=dict(color=get_hardcoded_colors(users_city['User_id']))
        ))
        fig_c2 = apply_powerbi_style(fig_c2, "User Count by City")
        st.plotly_chart(fig_c2, use_container_width=True, theme=None)
    with col3:
        ratings_city = city_stats_bars.sort_values(by='Rating_Count', ascending=False).head(10).sort_values(by='Rating_Count', ascending=True)
        fig_c3 = go.Figure(go.Bar(
            x=ratings_city['Rating_Count'], y=ratings_city['City'], orientation='h',
            text=ratings_city['Rating_Count'].apply(lambda x: f"{x}"), textposition='outside',
            marker=dict(color=get_hardcoded_colors(ratings_city['Rating_Count']))
        ))
        fig_c3 = apply_powerbi_style(fig_c3, "Rating Count by City")
        st.plotly_chart(fig_c3, use_container_width=True, theme=None)

with tab4:
    rkpi1, rkpi2, rkpi3, rkpi4 = st.columns(4)
    total_rests = df_orders['Restaurant_id'].nunique()
    avg_rating = df_orders['Rating'].mean()
    max_year = df_orders['order_date'].dt.year.max()
    curr_yr_sales = df_orders[df_orders['order_date'].dt.year == max_year]['Sales_amount'].sum()
    prev_yr_sales = df_orders[df_orders['order_date'].dt.year == (max_year - 1)]['Sales_amount'].sum()
    
    rkpi1.markdown(f"<div class='kpi-card'><div class='kpi-value'>{total_rests/1000:.1f}K</div><div class='kpi-label'>Restaurant Count</div></div>", unsafe_allow_html=True)
    rkpi2.markdown(f"<div class='kpi-card'><div class='kpi-value'>{avg_rating:.1f}</div><div class='kpi-label'>Avg Rating</div></div>", unsafe_allow_html=True)
    rkpi3.markdown(f"<div class='kpi-card'><div class='kpi-value'>{curr_yr_sales/1e6:.0f}M</div><div class='kpi-label'>Current YR Sales</div></div>", unsafe_allow_html=True)
    rkpi4.markdown(f"<div class='kpi-card'><div class='kpi-value'>{prev_yr_sales/1e6:.0f}M</div><div class='kpi-label'>Previous YR Sales</div></div>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        veg_nonveg = df_menu['Food_Type'].value_counts().reset_index()
        fig_r1 = go.Figure(go.Pie(
            labels=veg_nonveg['Food_Type'], values=veg_nonveg['count'], hole=0.6,
            marker=dict(colors=pbi_discrete_oranges), textinfo='percent+label', textposition='outside'
        ))
        fig_r1 = apply_powerbi_style(fig_r1, "Restaurant Veg-Nonveg")
        st.plotly_chart(fig_r1, use_container_width=True, theme=None)
        
    with col2:
        cuisine_price = df_menu.groupby('Cuisine')['Price'].sum().nlargest(10).reset_index().sort_values(by='Price', ascending=False)
        fig_r2 = go.Figure(go.Bar(
            x=cuisine_price['Cuisine'], y=cuisine_price['Price'],
            text=cuisine_price['Price'].apply(lambda x: f"{x/1e6:.1f}M" if x>=1000000 else f"{x/1000:.0f}K"), textposition='outside',
            marker=dict(color=get_hardcoded_colors(cuisine_price['Price']))
        ))
        fig_r2 = apply_powerbi_style(fig_r2, "Sum of Price by Cuisine", xtitle="Cuisine", ytitle="Total Price")
        st.plotly_chart(fig_r2, use_container_width=True, theme=None)
        
    col3, col4 = st.columns(2)
    with col3:
        rest_city = df_orders.groupby('City')['Restaurant_id'].nunique().nlargest(10).reset_index().sort_values(by='Restaurant_id', ascending=False)
        fig_r3 = go.Figure(go.Bar(
            x=rest_city['City'], y=rest_city['Restaurant_id'],
            text=rest_city['Restaurant_id'].apply(lambda x: f"{x/1000:.1f}K"), textposition='outside',
            marker=dict(color=get_hardcoded_colors(rest_city['Restaurant_id']))
        ))
        fig_r3 = apply_powerbi_style(fig_r3, "Restaurant Count by City")
        st.plotly_chart(fig_r3, use_container_width=True, theme=None)
    with col4:
        top_cuisine = df_menu['Cuisine'].value_counts().nlargest(5).reset_index()
        fig_r4 = go.Figure(go.Pie(
            labels=top_cuisine['Cuisine'], values=top_cuisine['count'], hole=0.6,
            marker=dict(colors=pbi_discrete_oranges), textinfo='percent+label', textposition='outside'
        ))
        fig_r4 = apply_powerbi_style(fig_r4, "Top 5 Cuisine")
        st.plotly_chart(fig_r4, use_container_width=True, theme=None)

with tab5:
    st.markdown("<h2 style='text-align: center; color: #ED7D31;'>Strategic Insights & Recommendations</h2>", unsafe_allow_html=True)
    
    st.markdown("<h4 style='color: #CCCCCC; margin-top: 20px;'>📊 Key Data Discoveries</h4>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    with col1:
        st.info("**Geographic Disconnect**\n\nTirupati drives the highest gross sales (42.5M), but Bikaner leads in sheer user volume (1.6K users). This reveals that Bikaner has high engagement with low order values, while Tirupati users place premium/high-value orders.")
    with col2:
        st.warning("**The Demographic Sweet Spot**\n\nMales outspend females by roughly 30% (545M vs 418M). Crucially, the 22-26 age bracket is the primary revenue engine, indicating a strong product-market fit with young working professionals.")
    with col3:
        st.success("**Menu Concentration**\n\nNorth Indian and Chinese cuisines completely dominate the restaurant supply side. Meanwhile, nearly 70% of all distinct menu items offered across the platform are strictly Vegetarian.")
        
    st.markdown("<hr style='border:1px solid #333;'>", unsafe_allow_html=True)
    st.markdown("<h4 style='color: #CCCCCC;'>🚀 Churn Mitigation Strategies</h4>", unsafe_allow_html=True)
    
    rc1, rc2, rc3 = st.columns(3)
    with rc1:
        st.error("""
        **🔴 High Risk Customers**
        *(Probability > 70%)*
        * **30% Discount Coupon** to aggressively win back dormant users
        * **Free Delivery** on next 3 orders
        * **Push Notification** featuring their most ordered cuisine
        * **Loyalty Bonus** added directly to wallet
        """)
        
    with rc2:
        st.warning("""
        **🟡 Medium Risk Customers**
        *(Probability 40% - 70%)*
        * **Personalized Recommendations** via email
        * **15% Discount** on new premium restaurants
        * **Limited-time Offers** to create urgency
        * **Feedback Surveys** to identify dissatisfaction
        """)
        
    with rc3:
        st.success("""
        **🟢 Low Risk Customers**
        *(Probability < 40%)*
        * **Promote Swiggy One** subscription for long-term lock-in
        * **Referral Campaigns** to leverage their loyalty
        * **Reward Points** for consistent ordering
        * **Exclusive Previews** of new platform features
        """)
        
    st.markdown("<hr style='border:1px solid #333;'>", unsafe_allow_html=True)
    st.markdown("<h4 style='color: #CCCCCC;'>🎯 Final Strategic Conclusion</h4>", unsafe_allow_html=True)
    st.markdown("""
    > <span style='color:#AAAAAA; font-size: 15px;'>This project and the underlying Random Forest Churn Model (81% Accuracy) prove:
    > 1. **Swiggy’s main problem is retention, not acquisition.** The historical data shows significant drop-offs after initial engagement.
    > 2. **Revenue depends heavily on a small segment.** The Top 10% of users generate nearly 70% of the gross platform revenue.
    > 3. **Predictive intervention can reduce churn significantly.** Our ML model accurately flags at-risk users based on active days and lifetime spend.
    > 4. **RFM segmentation combined with ML improves marketing precision.** Using Recency, Frequency, and Monetary value allows for surgical targeting rather than broad discount spraying.
    > 5. **A deployed churn model enables real-time business action.** Customer support and automated marketing can trigger interventions immediately.
    > 
    > **Reducing churn by even 10% across the high-risk segment could:**
    > * Stabilize month-over-month revenue
    > * Massively increase Customer Lifetime Value (LTV)
    > * Improve overall profitability by reducing CAC (Customer Acquisition Cost) dependency</span>
    """, unsafe_allow_html=True)

with tab6:
    st.markdown("<h3 style='color:#ED7D31;'>CHURN PREDICTOR</h3>", unsafe_allow_html=True)
    try:
        model = joblib.load('Processed_Data/best_churn_model.pkl')
        scaler = joblib.load('Processed_Data/scaler.pkl')
        col1, col2 = st.columns(2)
        with col1:
            in_orders = st.number_input("Total Orders", min_value=1, value=5)
            in_revenue = st.number_input("Total Lifetime Spend", min_value=0.0, value=2000.0)
            in_age = st.number_input("Customer Age", min_value=10, value=25)
        with col2:
            in_active_days = st.number_input("Active Days", min_value=0, value=30)
            in_gender = st.selectbox("Gender", ["Male", "Female", "Other"])
            
        if st.button("Predict Churn Risk", type='primary', use_container_width=True):
            avg_order = in_revenue / in_orders if in_orders > 0 else 0
            is_male = 1 if in_gender == "Male" else 0
            input_df = pd.DataFrame([{'total_orders': in_orders, 'total_revenue': in_revenue, 'active_days': in_active_days, 'avg_order_value': avg_order, 'Age': in_age, 'Gender_Male': is_male}])
            scaled_input = scaler.transform(input_df)
            prob = model.predict_proba(scaled_input)[0][1]
            if prob > 0.85:
                st.error(f"⚠️ **High Churn Risk!** (Probability: {prob:.1%})")
            elif prob > 0.70:
                st.warning(f"⚠️ **Moderate Churn Risk!** (Probability: {prob:.1%})")
            else:
                st.success(f"✅ **Safe Customer** (Probability: {prob:.1%})")
    except Exception as e:
        st.warning("Model not loaded properly.")