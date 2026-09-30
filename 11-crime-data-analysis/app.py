import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from scipy.stats import chi2_contingency

st.set_page_config(page_title="Reported Incident Explorer",page_icon="🗺️",layout="wide")
st.title("Reported Incident Explorer")
st.caption("Patterns in reported incidents—not claims about people, neighborhoods, or underlying crime prevalence.")


@st.cache_data
def demo(n=12000):
    rng=np.random.default_rng(19);start=pd.Timestamp("2021-01-01")
    date=start+pd.to_timedelta(rng.integers(0,1460,n),unit="D")+pd.to_timedelta(np.clip(rng.normal(15,6,n),0,23).astype(int),unit="h")
    category=rng.choice(["Theft report","Property damage","Burglary report","Assault report","Other report"],n,p=[.42,.2,.14,.12,.12])
    area=rng.choice(["North","Central","East","South","West"],n,p=[.18,.27,.2,.2,.15])
    lat=41.88+np.array([{"North":.07,"Central":0,"East":.01,"South":-.07,"West":0}[x] for x in area])+rng.normal(0,.018,n)
    lon=-87.68+np.array([{"North":.01,"Central":.03,"East":.06,"South":.02,"West":-.05}[x] for x in area])+rng.normal(0,.02,n)
    return pd.DataFrame({"date":date,"category":category,"area":area,"latitude":lat,"longitude":lon})


def normalize(d):
    aliases={"date":["date","datetime","incident_date","occurrence_date"],"category":["category","crime_type","primary_type","offense"],"area":["area","district","neighborhood","location"],"latitude":["latitude","lat"],"longitude":["longitude","lon","lng"]}
    rename={}
    for canonical,opts in aliases.items():
        found=next((c for c in d if c.lower() in opts),None)
        if found:rename[found]=canonical
    d=d.rename(columns=rename)
    if not {"date","category"}.issubset(d):raise ValueError("CSV needs date and category/crime_type columns.")
    d.date=pd.to_datetime(d.date,errors="coerce");return d.dropna(subset=["date","category"])


upload=st.sidebar.file_uploader("Optional public incident CSV",type="csv")
try:
    df=normalize(pd.read_csv(upload)) if upload else demo();source=f"Uploaded public-data file: {upload.name}" if upload else "Synthetic demonstration reports (not actual incidents)"
except Exception as exc:st.error(str(exc));st.stop()
df["year"]=df.date.dt.year;df["hour"]=df.date.dt.hour;df["weekday"]=pd.Categorical(df.date.dt.day_name(),["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"],ordered=True);df["weekend"]=np.where(df.date.dt.dayofweek>=5,"Weekend","Weekday")
types=sorted(df.category.astype(str).unique());chosen=st.sidebar.multiselect("Incident category",types,default=types)
years=sorted(df.year.unique());year_range=st.sidebar.select_slider("Year range",years,value=(years[0],years[-1]))
view=df[df.category.astype(str).isin(chosen)&df.year.between(*year_range)]
if "area" in df:
    areas=sorted(df.area.dropna().astype(str).unique());selected_areas=st.sidebar.multiselect("Area",areas,default=areas);view=view[view.area.astype(str).isin(selected_areas)]
tabs=st.tabs(["Trends","Time patterns","Map","Anomalies & tests","Methodology"])
with tabs[0]:
    st.info(source);a,b=st.columns(2);a.metric("Filtered reports",f"{len(view):,}");b.metric("Coverage",f"{view.date.min():%Y-%m-%d} to {view.date.max():%Y-%m-%d}")
    monthly=view.set_index("date").resample("MS").size().rename("Reports").reset_index()
    st.plotly_chart(px.line(monthly,x="date",y="Reports",title="Reported incidents by month"),width="stretch")
    st.plotly_chart(px.bar(view.category.value_counts().rename_axis("Category").reset_index(name="Reports"),x="Category",y="Reports",title="Category mix"),width="stretch")
with tabs[1]:
    c1,c2=st.columns(2);hour=view.groupby(["hour","weekend"],observed=True).size().rename("Reports").reset_index();c1.plotly_chart(px.line(hour,x="hour",y="Reports",color="weekend",title="Time-of-day patterns"),width="stretch")
    day=view.groupby("weekday",observed=False).size().rename("Reports").reset_index();c2.plotly_chart(px.bar(day,x="weekday",y="Reports",title="Reports by weekday"),width="stretch")
with tabs[2]:
    if {"latitude","longitude"}.issubset(view):
        mapped=view.dropna(subset=["latitude","longitude"]).sample(min(4000,len(view.dropna(subset=["latitude","longitude"]))),random_state=3)
        st.map(mapped,latitude="latitude",longitude="longitude",color="#e45756",size=8)
        st.caption("Point density reflects the available reports, collection system, and filtering—not population-adjusted risk.")
    else:st.info("Provide latitude and longitude columns to enable the map.")
with tabs[3]:
    daily=view.set_index("date").resample("D").size().rename("Reports");roll=daily.rolling(28,min_periods=14);z=(daily-roll.mean())/roll.std();anomalies=pd.DataFrame({"Reports":daily,"z_score":z}).dropna();anomalies=anomalies[anomalies.z_score.abs()>=3]
    st.write("Unusually high/low days relative to a trailing 28-day baseline:");st.dataframe(anomalies.sort_values("z_score",key=abs,ascending=False).head(30),width="stretch")
    table=pd.crosstab(view.weekend,view.category)
    if table.shape[0]>1 and table.shape[1]>1:
        chi2,p,dof,_=chi2_contingency(table);st.metric("Weekend/category chi-square p-value",f"{p:.4g}");st.caption("A small p-value indicates an association in this reporting sample; it does not establish cause or practical importance.")
with tabs[4]:
    st.markdown("""### Data and responsible interpretation
The default data is synthetic and exists only to demonstrate the workflow. Uploaded data should come from a documented government open-data portal; record the agency, download date, filters, and field definitions before analysis. This app parses dates, derives hour/day/year, counts reports, computes a rolling z-score, and uses a chi-square association test.

Reported incidents are not the same as underlying crime. Results can reflect reporting propensity, policing and classification practices, duplicate records, missing locations, administrative changes, population size, special events, and delayed entry. Raw counts are not population-adjusted. No map or difference here proves that a group or neighborhood is inherently dangerous.""")

