import streamlit as st
import pandas as pd
import qrcode
import math
import random
import numpy as np
import matplotlib.pyplot as plt
import uuid
from datetime import datetime
from sklearn.linear_model import LinearRegression
import time

# Page setup
st.set_page_config(page_title="SmartPark AI", layout="wide")

st.title("🚗 SmartPark AI – Intelligent Parking System")

# Load dataset
df = pd.read_csv("parking_data.csv")

# 🔥 SESSION STATE (NO LUB-DUB)
if "data" not in st.session_state:
    df["Price"] = df["Price"].apply(lambda x: random.randint(10,30))
    df["Available"] = df["Available"].apply(lambda x: max(0, x + random.randint(-2,2)))
    st.session_state.data = df
else:
    df = st.session_state.data

# Select location
location = st.selectbox("📍 Select Area", df["Location"])
selected = df[df["Location"] == location].iloc[0]

# Distance function
def distance(lat1, lon1, lat2, lon2):
    return math.sqrt((lat1-lat2)**2 + (lon1-lon2)**2)

# Calculate distance
df["Distance"] = df.apply(lambda row: distance(
    selected["lat"], selected["lon"],
    row["lat"], row["lon"]), axis=1)

# Best parking
available_df = df[df["Available"] > 0]
best = available_df.sort_values(by=["Distance","Price"]).iloc[0]

# UI cards
st.subheader("📍 Best Parking Spot")

col1, col2, col3 = st.columns(3)
col1.metric("Location", best["Location"])
col2.metric("Available Slots", int(best["Available"]))
col3.metric("Price/hr", f"₹{best['Price']}")

# 🏷️ Smart Insights
st.subheader("🏷️ Smart Insights")

closest = df.loc[df["Distance"].idxmin()]
cheapest = df.loc[df["Price"].idxmin()]

st.info(f"⭐ Recommended: {best['Location']}")
st.success(f"💰 Cheapest: {cheapest['Location']} (₹{cheapest['Price']})")
st.warning(f"⚡ Closest: {closest['Location']}")

# 🏆 Top 3
st.subheader("🏆 Top 3 Parking Options")

top3 = df.sort_values(by=["Distance","Price"]).head(3)

for _, row in top3.iterrows():
    st.write(f"{row['Location']} → {row['Available']} slots | ₹{row['Price']}")

# 🗺️ Map
st.subheader("🗺️ Parking Locations")
st.map(df[["lat", "lon"]])

# 📊 Availability Graph (LINE)
st.subheader("📊 Availability Graph")
fig, ax = plt.subplots()
ax.plot(df["Location"], df["Available"], marker='o')
plt.xticks(rotation=45)
st.pyplot(fig)

# 💰 Price Chart
st.subheader("💰 Price Chart")
fig2, ax2 = plt.subplots()
ax2.bar(df["Location"], df["Price"])
plt.xticks(rotation=45)
st.pyplot(fig2)

# 💰 Table
st.subheader("💰 Price Comparison")
st.dataframe(df[["Location","Price","Available"]].sort_values(by="Price"))

# 💳 Payment + Booking
st.subheader("💳 Booking & Payment")

if "payment_done" not in st.session_state:
    st.session_state.payment_done = False

if st.button("Proceed to Payment"):
    st.info("Processing Payment... ⏳")
    time.sleep(2)
    st.session_state.payment_done = True
    st.success("Payment Successful ✅")

if st.session_state.payment_done:
    if st.button("🎟️ Confirm Booking"):
        booking_id = str(uuid.uuid4())[:8]
        time_now = datetime.now().strftime("%H:%M:%S")

        info = f"ID: {booking_id}\nLocation: {best['Location']}\nTime: {time_now}"

        qr = qrcode.make(info)
        qr.save("qr.png")

        st.success("Booking Confirmed ✅")
        st.text(info)
        st.image("qr.png")

# 🤖 ML Prediction
st.subheader("🤖 Prediction")

time_data = np.array([[1],[2],[3],[4],[5],[6]])
slot_data = np.array([10,8,6,5,3,2])

model = LinearRegression()
model.fit(time_data, slot_data)

prediction = model.predict([[7]])

st.write(f"Next hour slots: {int(prediction[0])}")

if prediction[0] <= 3:
    st.error("⚠️ High demand expected!")
else:
    st.success("✅ Slots available")
