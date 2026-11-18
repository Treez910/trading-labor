"""Trading-Labor (Web-Version zum Teilen).

Ein Lernprojekt: Trendfolge-Strategie auf Krypto-Kursen, mit Backtest-Labor.
Kein Journal, kein echtes Geld, keine Anlageberatung.
"""

from pathlib import Path

import ccxt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="Trading-Labor", page_icon="📈", layout="wide")

# --- Einstellungen -----------------------------------------------------------
COINS = {"BTC": "Bitcoin", "ETH": "Ethereum"}
SCHNELL, LANGSAM = 40, 250
STARTKAPITAL = 100.0
GEBUEHR, SLIPPAGE = 0.10, 0.02
DATEN = Path(__file__).parent / "daten"
if not DATEN.exists():          # Dateien liegen evtl. direkt im Hauptordner
    DATEN = Path(__file__).parent
# Live-Kurse: erste Börse, die antwortet, wird benutzt
QUELLEN = [("kraken", "{c}/USD"), ("bitstamp", "{c}/USD"), ("binance", "{c}/USDT")]
GRUEN, ROT, ORANGE, BLAU, GRAU = "#22c55e", "#ef4444", "#f59e0b", "#3b82f6", "#94a3b8"

st.markdown("""
<style>
.block-container {padding-top: 2rem;}
.signal {border-radius: 14px; padding: 18px 20px; margin-bottom: 12px;}
.signal h2 {margin: 0; font-size: 1.9rem;}
.signal p {margin: 4px 0 0; opacity: .85;}
</style>
""", unsafe_allow_html=True)


# --- Rechenkern (wie im Labor) ----------------------------------------------
def ma_kreuzung(df, schnell, langsam):
    return (df["close"].rolling(schnell).mean()
            > df["close"].rolling(langsam).mean()).astype(int)


def backtest(df, signal, gebuehr, slippage, startkapital):
    position = signal.shift(1).fillna(0)
    rendite_kerze = df["open"].shift(-1) / df["open"] - 1
    wechsel = position.diff().abs().fillna(position.abs())
    netto = (position * rendite_kerze - wechsel * (gebuehr + slippage) / 100).fillna(0)
    return pd.DataFrame({"position": position, "rendite": netto,
                         "konto": startkapital * (1 + netto).cumprod(), "wechsel": wechsel})


def auswertung(erg):
    konto = erg["konto"]
    pos = erg["position"]
    block = (pos.diff() != 0).cumsum()
    trades = [(1 + t["rendite"]).prod() - 1 for _, t in erg[pos == 1].groupby(block[pos == 1])]
    return {
        "end": konto.iloc[-1],
        "drawdown": ((konto / konto.cummax() - 1) * 100).min(),
        "trades": len(trades),
        "treffer": (np.mean([t > 0 for t in trades]) * 100) if trades else 0.0,
    }


# --- Daten ------------------------------------------------------------------
@st.cache_data(ttl=900, show_spinner="Lade aktuelle Kurse ...")
def live_kerzen(coin: str):
    fehler = []
    for name, muster in QUELLEN:
        try:
            boerse = getattr(ccxt, name)({"enableRateLimit": True, "timeout": 15000})
            roh = boerse.fetch_ohlcv(muster.format(c=coin), "4h", limit=720)
            df = pd.DataFrame(roh, columns=["zeit", "open", "high", "low", "close", "volumen"])
            df["zeit"] = pd.to_datetime(df["zeit"], unit="ms", utc=True)
            df = df.drop_duplicates("zeit").set_index("zeit").sort_index()
            df = df[df.index + pd.Timedelta("4h") <= pd.Timestamp.now(tz="UTC")]  # nur fertige Kerzen
            if len(df) < LANGSAM + 20:
                raise ValueError(f"nur {len(df)} Kerzen")
            df["ma_s"] = df["close"].rolling(SCHNELL).mean()
            df["ma_l"] = df["close"].rolling(LANGSAM).mean()
            return df.dropna(), name.capitalize()
        except Exception as e:
            fehler.append(f"{name}: {e}")
    raise RuntimeError(" | ".join(fehler))


@st.cache_data(show_spinner="Lade historische Daten ...")
def lange_daten(coin: str):
    datei = DATEN / f"{coin}_USDT_4h_lang.csv"
    return pd.read_csv(datei, index_col="zeit", parse_dates=True) if datei.exists() else None


@st.cache_data(show_spinner="Rechne 150 Varianten durch ...")
def heatmap_werte(coin: str, gebuehr: float):
    df = lange_daten(coin)
    halten = backtest(df, pd.Series(1, index=df.index), gebuehr, SLIPPAGE, STARTKAPITAL)
    basis = halten["konto"].iloc[-1]
    schnell_w, langsam_w = list(range(10, 101, 10)), list(range(50, 401, 25))
    werte = pd.DataFrame(index=schnell_w, columns=langsam_w, dtype=float)
    for s in schnell_w:
        for l in langsam_w:
            if s < l:
                erg = backtest(df, ma_kreuzung(df, s, l), gebuehr, SLIPPAGE, STARTKAPITAL)
                werte.loc[s, l] = erg["konto"].iloc[-1] / basis
    return werte


def layout(fig, hoehe=420):
    fig.update_layout(template="plotly_dark", height=hoehe, margin=dict(l=10, r=10, t=30, b=10),
                      legend=dict(orientation="h", y=1.02, yanchor="bottom", x=0),
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    return fig


def zahl(x, nachkomma=0):
    return f"{x:,.{nachkomma}f}".replace(",", "X").replace(".", ",").replace("X", ".")


# --- Kopf -------------------------------------------------------------------
st.title("📈 Trading-Labor")
st.caption(f"Trendfolge-Strategie MA {SCHNELL}/{LANGSAM} auf 4-Stunden-Kerzen · "
           "ein Lernprojekt mit echten Marktdaten")
st.info("**Kein Finanzrat.** Das hier ist ein Lernprojekt, keine Anlageberatung und keine "
        "Kaufempfehlung. Ergebnisse aus der Vergangenheit garantieren nichts für die Zukunft. "
        "Krypto kann stark fallen, investier nie Geld, das du nicht verlieren kannst.", icon="⚠️")

tab_jetzt, tab_spiel, tab_charts, tab_labor, tab_kosten, tab_ueber = st.tabs(
    ["🏠 Jetzt", "🎮 Schlag die Strategie", "📊 Charts", "🧪 Labor", "💸 Gebühren-Rechner",
     "ℹ️ Wie funktioniert das?"])

# --- Jetzt ------------------------------------------------------------------
with tab_jetzt:
    quellen = set()
    for spalte, (coin, name) in zip(st.columns(len(COINS)), COINS.items()):
        with spalte:
            st.subheader(name)
            try:
                df, quelle = live_kerzen(coin)
                quellen.add(quelle)
            except Exception as fehler:
                st.error(f"Kurse gerade nicht erreichbar. ({fehler})")
                continue
            z = df.iloc[-1]
            drin = bool(z["ma_s"] > z["ma_l"])
            abstand = (z["ma_s"] / z["ma_l"] - 1) * 100
            farbe = GRUEN if drin else ROT
            text = "DRIN · Trend aufwärts" if drin else "DRAUSSEN · Trend abwärts"

            zustand = (df["ma_s"] > df["ma_l"])
            wechsel = zustand[zustand != zustand.iloc[-1]]
            if wechsel.empty:
                seit = "seit über 2 Monaten"
            else:
                tage = (df.index[-1] - wechsel.index[-1]).days
                seit = f"seit {tage} Tag{'en' if tage != 1 else ''} ({wechsel.index[-1]:%d.%m.%Y})"
            st.markdown(f"<div class='signal' style='background:{farbe}22;border:2px solid {farbe}'>"
                        f"<h2 style='color:{farbe}'>{text}</h2><p>Signal {seit}</p></div>",
                        unsafe_allow_html=True)

            vor_24h = df["close"].iloc[-7]
            m1, m2 = st.columns(2)
            m1.metric("Kurs", f"{zahl(z['close'])} $", f"{(z['close'] / vor_24h - 1) * 100:+.1f} % (24 h)")
            m2.metric(f"Abstand MA {SCHNELL}/{LANGSAM}", f"{abstand:+.1f} %",
                      help="Liegt der Wert über 0 %, ist die Strategie investiert. "
                           "Fällt er unter 0 %, steigt sie aus.")
            st.progress(min(max((abstand + 20) / 40, 0.0), 1.0),
                        text="◀ draußen · · · 0 % · · · drin ▶")
    if quellen:
        st.caption(f"Kursdaten: {', '.join(sorted(quellen))} · letzte abgeschlossene "
                   "4-Stunden-Kerze · wird alle 15 Minuten neu geladen")

# --- Charts -----------------------------------------------------------------
with tab_charts:
    c1, c2 = st.columns([1, 2])
    coin = c1.radio("Coin", list(COINS), format_func=COINS.get, horizontal=True, key="c_coin")
    tage = c2.slider("Zeitraum (Tage)", 14, 75, 60)
    try:
        df, _ = live_kerzen(coin)
        df = df[df.index >= df.index[-1] - pd.Timedelta(days=tage)]
        fig = go.Figure()
        fig.add_trace(go.Candlestick(x=df.index, open=df["open"], high=df["high"], low=df["low"],
                                     close=df["close"], name="Kurs",
                                     increasing_line_color=GRUEN, decreasing_line_color=ROT))
        fig.add_trace(go.Scatter(x=df.index, y=df["ma_s"], name=f"MA {SCHNELL}",
                                 line=dict(color=ORANGE, width=2)))
        fig.add_trace(go.Scatter(x=df.index, y=df["ma_l"], name=f"MA {LANGSAM}",
                                 line=dict(color=BLAU, width=2)))
        fig.update_layout(xaxis_rangeslider_visible=False)
        st.markdown(f"**{COINS[coin]}: Kurs und Durchschnitte**")
        st.plotly_chart(layout(fig, 520), width="stretch", key="chart_kurs")
        st.caption("Orange über Blau = Trend aufwärts (drin). Orange unter Blau = Trend abwärts (draußen).")
    except Exception as fehler:
        st.error(f"Kurse gerade nicht erreichbar. ({fehler})")

# --- Labor ------------------------------------------------------------------
with tab_labor:
    st.subheader("🧪 Backtest-Labor")
    st.caption("Teste eigene Einstellungen auf rund 9 Jahren historischer 4-Stunden-Daten. "
               "Startkapital jeweils 100 €.")
    l1, l2, l3, l4 = st.columns(4)
    coin = l1.selectbox("Coin", list(COINS), format_func=COINS.get, key="l_coin")
    schnell = l2.slider("Schneller Durchschnitt", 10, 100, SCHNELL, 5)
    langsam = l3.slider("Langsamer Durchschnitt", 50, 400, LANGSAM, 25)
    gebuehr = l4.number_input("Gebühr pro Seite (%)", 0.0, 2.0, GEBUEHR, 0.05)
    zeitraum = st.radio("Zeitraum", ["Alles", "Ab 2020", "Ab 2023", "Letzte 12 Monate"],
                        horizontal=True)

    df = lange_daten(coin)
    if df is None:
        st.warning("Historische Daten fehlen im Ordner daten.")
    elif schnell >= langsam:
        st.warning("Der schnelle Durchschnitt muss kleiner sein als der langsame.")
    else:
        signal = ma_kreuzung(df, schnell, langsam)
        start = {"Alles": df.index[0], "Ab 2020": pd.Timestamp("2020-01-01", tz="UTC"),
                 "Ab 2023": pd.Timestamp("2023-01-01", tz="UTC"),
                 "Letzte 12 Monate": df.index[-1] - pd.Timedelta(days=365)}[zeitraum]
        teil, sig = df[df.index >= start], signal[signal.index >= start]
        strat = backtest(teil, sig, gebuehr, SLIPPAGE, STARTKAPITAL)
        halten = backtest(teil, pd.Series(1, index=teil.index), gebuehr, SLIPPAGE, STARTKAPITAL)
        a, h = auswertung(strat), auswertung(halten)

        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Endkapital Strategie", f"{zahl(a['end'])} €",
                  f"{'+' if a['end'] >= h['end'] else '−'}{zahl(abs(a['end'] - h['end']))} € vs. Halten")
        k2.metric("Größter Rückgang", f"{a['drawdown']:.0f} %")
        k2.caption(f"Beim Halten: {h['drawdown']:.0f} %")
        k3.metric("Trades", a["trades"])
        k4.metric("Trefferquote", f"{a['treffer']:.0f} %")

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=strat.index, y=strat["konto"], name=f"MA {schnell}/{langsam}",
                                 line=dict(color=GRUEN, width=2)))
        fig.add_trace(go.Scatter(x=halten.index, y=halten["konto"], name="Nur halten",
                                 line=dict(color=GRAU, width=1.5)))
        fig.update_yaxes(type="log", title="Kontostand (log. Skala)")
        st.markdown(f"**Was wäre aus 100 € geworden?** (Endkapital Halten: {zahl(h['end'])} €)")
        st.plotly_chart(layout(fig, 450), width="stretch", key="chart_labor")

        with st.expander("🔥 Nachbarschaftstest: Sind die Nachbarwerte auch gut?"):
            st.caption("Jedes Feld zeigt das Endkapital einer Einstellung geteilt durch Halten "
                       "(über alle Jahre). Über 1 = besser, unter 1 = schlechter. "
                       "Eine ganze grüne Gegend ist ein gutes Zeichen, ein einzelnes grünes Feld eher Zufall.")
            werte = heatmap_werte(coin, gebuehr)
            log = np.log2(werte.astype(float))
            grenze = float(np.nanmax(np.abs(log.values)))
            hm = go.Figure(go.Heatmap(
                z=log.values, x=[str(c) for c in werte.columns], y=[str(i) for i in werte.index],
                text=werte.round(2).values, texttemplate="%{text}", colorscale="RdYlGn",
                zmid=0, zmin=-grenze, zmax=grenze, showscale=False,
                hovertemplate="MA %{y}/%{x}: %{text}x Halten<extra></extra>"))
            hm.update_xaxes(title="Langsamer Durchschnitt", type="category")
            hm.update_yaxes(title="Schneller Durchschnitt", type="category", autorange="reversed")
            st.plotly_chart(layout(hm, 480), width="stretch", key="heatmap")

        st.warning("⚠️ Vorsicht Overfitting: Die Einstellung, die rückblickend am besten war, "
                   "ist nicht automatisch die beste für die Zukunft.")

# --- Spiel: Schlag die Strategie --------------------------------------------
SPIEL_TAGE, VORLAUF = 30, 45
KOSTEN = (GEBUEHR + SLIPPAGE) / 100


@st.cache_data
def tageskerzen(coin: str):
    df = lange_daten(coin)
    if df is None:
        return None
    t = df.resample("1D").agg({"open": "first", "high": "max", "low": "min",
                               "close": "last"}).dropna()
    t["ma_s"] = t["close"].rolling(7).mean()     # ≈ MA 40 auf 4 h
    t["ma_l"] = t["close"].rolling(42).mean()    # ≈ MA 250 auf 4 h
    return t.dropna()


def neue_runde():
    rng = np.random.default_rng()
    coin = str(rng.choice(list(COINS)))
    t = tageskerzen(coin)
    if t is None:
        st.session_state.pop("spiel", None)
        return
    start = int(rng.integers(VORLAUF, len(t) - SPIEL_TAGE))
    st.session_state["spiel"] = {"coin": coin, "start": start, "schritt": 0,
                                 "pos": 0, "du": [100.0], "positionen": []}


def spiel_schritt(position: int, tage: int = 1):
    g = st.session_state["spiel"]
    t = tageskerzen(g["coin"])
    for _ in range(tage):
        if g["schritt"] >= SPIEL_TAGE:
            break
        i = g["start"] + g["schritt"]
        wert = g["du"][-1]
        if position != g["pos"]:
            wert *= 1 - KOSTEN
            g["pos"] = position
        if position == 1:
            wert *= t["close"].iloc[i] / t["close"].iloc[i - 1]
        g["du"].append(wert)
        g["positionen"].append(position)
        g["schritt"] += 1


def regel_verlauf(t, start):
    wert, pos, verlauf = 100.0, 0, [100.0]
    for k in range(SPIEL_TAGE):
        i = start + k
        signal = int(t["ma_s"].iloc[i - 1] > t["ma_l"].iloc[i - 1])
        if signal != pos:
            wert *= 1 - KOSTEN
            pos = signal
        if pos == 1:
            wert *= t["close"].iloc[i] / t["close"].iloc[i - 1]
        verlauf.append(wert)
    return verlauf


with tab_spiel:
    st.subheader("🎮 Schlag die Strategie")
    st.caption(f"Ein echter, zufälliger Ausschnitt aus den letzten Jahren, ohne Datum und "
               f"ohne Preisangabe. Du entscheidest {SPIEL_TAGE} Tage lang jeden Tag: drin oder "
               "draußen? Jeder Wechsel kostet Gebühren. Am Ende siehst du, ob du die Regel "
               "und einfaches Halten geschlagen hast.")
    if "spiel" not in st.session_state:
        neue_runde()
    g = st.session_state.get("spiel")
    if g is None:
        st.warning("Historische Daten fehlen.")
    else:
        t = tageskerzen(g["coin"])
        s0 = g["start"]
        basis = t["close"].iloc[s0 - 1]
        sicht = t.iloc[s0 - VORLAUF: s0 + g["schritt"]]
        x = list(range(-VORLAUF + 1, g["schritt"] + 1))
        norm = 100 / basis

        fig = go.Figure()
        fig.add_trace(go.Candlestick(x=x, open=sicht["open"] * norm, high=sicht["high"] * norm,
                                     low=sicht["low"] * norm, close=sicht["close"] * norm,
                                     name="Kurs", increasing_line_color=GRUEN,
                                     decreasing_line_color=ROT))
        for k, p in enumerate(g["positionen"], start=1):
            if p == 1:
                fig.add_vrect(x0=k - 0.5, x1=k + 0.5, fillcolor=GRUEN, opacity=0.12, line_width=0)
        fig.add_vline(x=0.5, line_dash="dot", line_color=GRAU)
        fig.update_layout(xaxis_rangeslider_visible=False, showlegend=False)
        fig.update_xaxes(title="Tag", range=[-VORLAUF + 0.5, SPIEL_TAGE + 0.5])
        fig.update_yaxes(title="Kurs (Start = 100)")
        st.plotly_chart(layout(fig, 430), width="stretch", key="spiel_chart")
        st.caption("Links der gepunkteten Linie: Vorgeschichte. Grün hinterlegt: Tage, an denen du drin warst.")

        if g["schritt"] < SPIEL_TAGE:
            s1, s2, s3 = st.columns(3)
            s1.metric("Tag", f"{g['schritt'] + 1} von {SPIEL_TAGE}")
            s2.metric("Dein Konto", f"{g['du'][-1]:.2f} €")
            s3.metric("Du bist gerade", "DRIN" if g["pos"] else "DRAUSSEN")
            b1, b2, b3 = st.columns(3)
            if b1.button("📈 Morgen drin sein", type="primary", width="stretch"):
                spiel_schritt(1); st.rerun()
            if b2.button("💵 Morgen draußen sein", width="stretch"):
                spiel_schritt(0); st.rerun()
            if b3.button("⏩ 5 Tage weiter wie bisher", width="stretch"):
                spiel_schritt(g["pos"], 5); st.rerun()
        else:
            regel = regel_verlauf(t, s0)
            halten = [100.0] + [100 * (1 - KOSTEN) * t["close"].iloc[s0 + k] / basis
                                for k in range(SPIEL_TAGE)]
            du, r, h = g["du"][-1], regel[-1], halten[-1]
            if du > max(r, h):
                st.success(f"🏆 Stark! Du hast beide geschlagen: {du:.2f} €")
            elif du > r or du > h:
                st.info(f"👍 Nicht schlecht! Du hast einen von beiden geschlagen: {du:.2f} €")
            else:
                st.error(f"😅 Diesmal nicht. Du: {du:.2f} €")
            e1, e2, e3 = st.columns(3)
            e1.metric("Du", f"{du:.2f} €", f"{du - 100:+.2f} €")
            e2.metric(f"Regel (MA-Trendfolge)", f"{r:.2f} €", f"{r - 100:+.2f} €")
            e3.metric("Nur halten", f"{h:.2f} €", f"{h - 100:+.2f} €")
            wechsel = sum(1 for a, b in zip([0] + g["positionen"], g["positionen"]) if a != b)
            st.caption(f"Aufgelöst: Das war **{COINS[g['coin']]}** vom "
                       f"{t.index[s0]:%d.%m.%Y} bis {t.index[s0 + SPIEL_TAGE - 1]:%d.%m.%Y}. "
                       f"Du hast {wechsel}× gewechselt und dafür rund "
                       f"{wechsel * KOSTEN * 100:.2f} % Gebühren gezahlt.")
            vf = go.Figure()
            vf.add_trace(go.Scatter(y=g["du"], name="Du", line=dict(color=ORANGE, width=3)))
            vf.add_trace(go.Scatter(y=regel, name="Regel", line=dict(color=GRUEN, width=2)))
            vf.add_trace(go.Scatter(y=halten, name="Halten", line=dict(color=GRAU, dash="dash")))
            vf.update_xaxes(title="Tag")
            st.plotly_chart(layout(vf, 320), width="stretch", key="spiel_ergebnis")

            bilanz = st.session_state.setdefault("bilanz", {"runden": 0, "siege": 0})
            if not g.get("gezaehlt"):
                bilanz["runden"] += 1
                bilanz["siege"] += int(du > r)
                g["gezaehlt"] = True
            st.caption(f"Deine Bilanz in dieser Sitzung: {bilanz['siege']} von "
                       f"{bilanz['runden']} Runden besser als die Regel.")
            if st.button("🔄 Neue Runde", type="primary"):
                neue_runde(); st.rerun()
        st.caption("Über einzelne 30-Tage-Runden entscheidet viel Zufall, auch bei der Regel. "
                   "Spiel mehrere Runden, dann wird es interessant.")

# --- Gebühren-Rechner -------------------------------------------------------
with tab_kosten:
    st.subheader("💸 Was kostet mich mein Trading wirklich?")
    st.caption("Gebühren sind unsichtbar, weil sie pro Trade klein aussehen. "
               "Hier siehst du, was sich über ein Jahr ansammelt.")
    g1, g2, g3, g4 = st.columns(4)
    konto = g1.number_input("Kontogröße (€)", 10.0, 1_000_000.0, 100.0, 10.0)
    pro_woche = g2.number_input("Trades pro Woche", 0, 500, 10, 1,
                                help="Ein Trade = einmal kaufen und wieder verkaufen.")
    geb = g3.number_input("Gebühr pro Kauf/Verkauf (%)", 0.0, 5.0, 0.10, 0.05)
    spread = g4.number_input("Spread/Slippage (%)", 0.0, 5.0, 0.05, 0.01,
                             help="Unterschied zwischen Kauf- und Verkaufskurs. Bei manchen "
                                  "Apps steckt hier ein Großteil der Kosten.")
    pro_trade = 2 * geb + spread
    trades_jahr = pro_woche * 52
    rest = konto * (1 - pro_trade / 100) ** trades_jahr
    verlust = konto - rest
    r1, r2, r3 = st.columns(3)
    r1.metric("Kosten pro Trade", f"{pro_trade:.2f} %")
    r2.metric("Trades pro Jahr", f"{trades_jahr:,}".replace(",", "."))
    r3.metric("Kosten nach 1 Jahr", f"{zahl(verlust, 2)} €",
              f"{verlust / konto * 100:.0f} % deines Kontos", delta_color="inverse")
    st.markdown(f"Um nur die Gebühren wieder reinzuholen, muss **jeder einzelne Trade im Schnitt "
                f"mehr als {pro_trade:.2f} % Gewinn** bringen. Nach einem Jahr wären ohne "
                f"Gewinne von {zahl(konto, 2)} € nur noch **{zahl(rest, 2)} €** übrig.")
    wochen = np.arange(0, 53)
    kurve = konto * (1 - pro_trade / 100) ** (wochen * pro_woche)
    kf = go.Figure(go.Scatter(x=wochen, y=kurve, fill="tozeroy", line=dict(color=ROT, width=2),
                              name="Konto"))
    kf.update_xaxes(title="Woche"); kf.update_yaxes(title="Kontostand (€)", rangemode="tozero")
    st.markdown("**So schmilzt das Konto nur durch Kosten, ohne einen einzigen Verlusttrade:**")
    st.plotly_chart(layout(kf, 320), width="stretch", key="kosten_chart")
    st.caption("Vereinfachte Rechnung: gleich große Trades mit dem ganzen Konto, ohne Gewinne oder "
               "Verluste. Die Gebühren deiner Börse findest du in deren Preisverzeichnis.")

# --- Über -------------------------------------------------------------------
with tab_ueber:
    st.markdown(f"""
### Die Idee in einem Satz
Investiert sein, solange der Kurs im mittelfristigen Aufwärtstrend ist, und aussteigen, wenn der Trend kippt.

### Die Regel
- **MA {SCHNELL}:** Durchschnittskurs der letzten {SCHNELL} Vier-Stunden-Kerzen (etwa 7 Tage)
- **MA {LANGSAM}:** Durchschnittskurs der letzten {LANGSAM} Vier-Stunden-Kerzen (etwa 42 Tage)
- Liegt MA {SCHNELL} über MA {LANGSAM}: **drin**. Liegt er darunter: **draußen**.
- Das ergibt nur etwa 5 bis 10 Trades pro Jahr.

### Was wir im Labor herausgefunden haben
1. **Gebühren sind der größte Gegner.** Strategien auf 15-Minuten-Kerzen haben durch hunderte Trades im Jahr fast alles verloren, obwohl die Idee ohne Gebühren nicht schlecht war.
2. **Ein einzelnes Jahr sagt fast nichts.** Der Gewinner aus einem Jahr war über 9 Jahre nur Mittelmaß.
3. **Langsames Trendfolgen hat über viele Jahre gehalten**, vor allem weil es große Crashs wie 2018 und 2022 zu großen Teilen vermieden hat. In starken Aufwärtsjahren hinkt es dafür oft hinterher.
4. **Der Vorteil ist robust, aber nicht riesig:** Viele benachbarte Einstellungen funktionieren ähnlich, und die Strategie schlägt die meisten Zufallsstrategien. Getestet wurde aber nur mit Bitcoin und Ethereum, also zwei Coins, die bis heute überlebt haben.

### Was die Tests nicht abbilden
Steuern, Ausfälle von Börsen, extreme Kurssprünge zwischen zwei Kerzen und vor allem: ob man selbst die Nerven hat, jedem Signal zu folgen.
""")
    st.caption("Historische Daten: Binance, 4-Stunden-Kerzen ab Ende 2017. "
               "Gebühr im Test: 0,10 % pro Kauf/Verkauf plus 0,02 % Slippage.")
