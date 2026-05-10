#!/usr/bin/env python3
import sqlite3
import pandas as pd
import streamlit as st
import plotly.express as px

DB_PATH = 'data/spy_truth.db'

st.title('SharpEdge SPY Tape Map')

conn = sqlite3.connect(DB_PATH)

query = '''
SELECT
    price,
    size,
    side,
    ts
FROM tape_ticks
ORDER BY ts DESC
LIMIT 500
'''

df = pd.read_sql(query, conn)

if df.empty:
    st.warning('No tape data found.')
    st.stop()

fig = px.scatter(
    df,
    x='ts',
    y='price',
    size='size',
    color='side',
    title='Recent SPY Tape Activity'
)

st.plotly_chart(fig, use_container_width=True)

st.dataframe(df.head(100))
