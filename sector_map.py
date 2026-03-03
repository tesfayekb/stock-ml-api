"""Sector classification map — mirrors edge function sector assignments."""

SECTOR_MAP: dict[str, str] = {}


def _sm(sector: str, tickers: list[str]):
    for t in tickers:
        SECTOR_MAP[t] = sector


_sm("Technology", ["AAPL","MSFT","GOOGL","GOOG","NVDA","META","AVGO","ORCL","CRM","ADBE","AMD","CSCO","ACN","INTC","IBM","INTU","NOW","QCOM","TXN","AMAT","ADI","LRCX","MU","KLAC","SNPS","CDNS","MRVL","FTNT","PANW","CRWD","NXPI","ON","MPWR","ANSS","KEYS","GEN","CTSH","IT","FSLR","EPAM","ENPH","SEDG","AKAM","FFIV","JNPR","SWKS","QRVO","WDC","PLTR","SOFI","HOOD","SNAP","PINS","ZM","ROKU","TWLO","SQ","COIN","AFRM","UPST","SHOP","NET","SNOW","DDOG","MDB","ZS","OKTA","S","MNDY","TEAM","ATLSY","TTD","TRADE","PATH","AI","BBAI","BIGC","CFLT","GTLB","HUBS","PCOR","ESTC","TYL","TOST","DLO","FOUR","GFS","WOLF","SMCI","ARM","VRT","GEV","IONQ","RGTI","QUBT","KULR","ALAB","OKLO","SMR","APLD","RMBS","SMTC","ONTO","GLOB","LSCC","PI","CGNX","MASI","CYBR","MANH","BILL","HQY","TNET","PAYC","CALX","FORM","DIOD","POWI","COHU","ICHR","AEIS","CSGS","HLIT","AMBA","BRZE","DOCN","VERX","SPSC","VRNT","PRGS","DUOL"])
_sm("Healthcare", ["UNH","JNJ","LLY","ABBV","MRK","TMO","ABT","PFE","DHR","AMGN","ELV","BMY","ISRG","GILD","MDT","CI","SYK","VRTX","REGN","ZTS","BSX","BDX","HUM","IDXX","IQV","EW","A","DXCM","MTD","WAT","BAX","ALGN","HOLX","TFX","TECH","MOH","CNC","HSIC","XRAY","OGN","DVA","VTRS","CTLT","INCY","MRNA","BIIB","ILMN","PKI","LH","DGX","PCVX","NBIX","LNTH","BPMC","ALNY","HALO","RGEN","ENSG","CHE","EHC","EXEL","MEDP","GMED","NVST","MMSI","HAE","CPRX","TMDX","AGIO","XNCR","SMMT","RXRX","TEM","PDCO","OMCL","CORT"])
_sm("Financials", ["JPM","V","MA","BAC","WFC","GS","MS","BLK","SCHW","AXP","C","SPGI","CME","ICE","MCO","AON","MMC","CB","PGR","MET","AIG","TRV","AFL","PRU","ALL","CINF","FITB","MTB","HBAN","RF","KEY","CFG","ZION","CMA","NDAQ","MSCI","CBOE","RJF","FRC","BEN","TROW","IVZ","WRB","GL","RE","L","AIZ","LNC","MKTX","OWL","IBKR","SF","EWBC","GBCI","FHN","WTFC","UMBF","SBCF","PB","WAL","BOH","CADE","CVBF","FNB","HWC","OZK","SSB","SNV","ASB","BKU","CUB","FBP","FCNCA","FFIN","HOPE","PIPR","PNFP","HLI","SCI","RNR","DKNG","SIGI","NMIH","TBBK","TCBI","WAFD","WSBC","WSFS","BANR","COLB","DCOM","FFBC","FULT","GABC","HAFC","HTLF","IBTX","INDB","MBWM","NBTB","NWBI","ONB","PEBO","PFBC","PPBI","RNST","SBSI","SFNC","TRMK","UBSI","UCBI","VBTX","WABC","EFSC","NBHC","OFG","NU","PAYO"])
_sm("Consumer Discretionary", ["AMZN","TSLA","HD","MCD","NKE","SBUX","LOW","TJX","BKNG","CMG","ABNB","MAR","HLT","ORLY","AZO","ROST","DHI","LEN","PHM","NVR","GPC","POOL","BBY","APTV","GRMN","EBAY","ETSY","DRI","YUM","MGM","WYNN","CZR","LVS","RCL","CCL","NCLH","F","GM","TSCO","KMX","AAP","BWA","SEE","HAS","NWL","WHR","PVH","RL","TPR","VFC","PENN","DECK","WSM","LULU","BURL","FND","SFM","CROX","SKX","RIVN","LCID","DASH","RBLX","U","MELI","SE","GRAB","CPNG","BABA","JD","PDD","BIDU","NIO","XPEV","LI","CART","BIRK","CAVA","BOOT","CARG","SHOO","FIZZ","GMS"])
_sm("Consumer Staples", ["PG","KO","PEP","COST","WMT","PM","MO","MDLZ","CL","KMB","EL","STZ","GIS","SJM","K","HSY","CPB","HRL","MKC","CHD","CAG","TSN","KHC","BG","ADM","TAP","MNST","KDP","CLX","SYY","CELH","LANC"])
_sm("Industrials", ["CAT","DE","UNP","UPS","HON","GE","BA","RTX","LMT","MMM","GD","NOC","TT","PH","ROK","EMR","ETN","ITW","FAST","CTAS","PCAR","ODFL","CSX","NSC","FDX","DAL","UAL","LUV","ALK","JBHT","XYL","WM","RSG","VRSK","IR","DOV","SWK","SNA","RHI","PWR","SAIA","WFRD","RBC","FLR","JBL","KBR","CACI","AZEK","CSWI","ESAB","WTS","SITE","TTC","AOS","GNRC","TREX","UFPI","WEX","EXPO","AVNT","KTOS","INST","AIT","WOR","RXO","AAON","JOBY","ACHR","LILM","EVTL","BLDE","RKLB","ASTS","SPCE","LUNR","MNTS","RDW","GRRR"])
_sm("Energy", ["XOM","CVX","COP","EOG","SLB","MPC","VLO","PSX","PXD","DVN","OXY","HAL","FANG","HES","BKR","TRGP","WMB","KMI","OKE","CTRA"])
_sm("Utilities", ["NEE","DUK","SO","D","AEP","SRE","EXC","XEL","WEC","ES","ED","PEG","AWK","ATO","CMS","CNP","NI","EVRG","FE","PPL","MGEE","CWEN"])
_sm("Real Estate", ["PLD","AMT","CCI","EQIX","PSA","DLR","O","WELL","SPG","AVB","EQR","VTR","ARE","MAA","UDR","ESS","PEAK","KIM","REG","BXP","COOP"])
_sm("Materials", ["LIN","APD","SHW","ECL","FCX","NEM","NUE","VMC","MLM","DOW","DD","PPG","CE","ALB","FMC","CF","MOS","IFF","EMN","WDFC","BCPC"])
_sm("Communication Services", ["DIS","NFLX","CMCSA","T","VZ","CHTR","TMUS","EA","TTWO","WBD","PARA","FOX","FOXA","NWS","NWSA","MTCH","IPG","OMC","LYV","BRBR"])
_sm("Crypto/Blockchain", ["IREN","CLSK","MARA","RIOT","HUT","BITF","WULF","CIFR","CORZ"])


def get_sector(ticker: str) -> str:
    """Return sector for a ticker, or 'unknown' if not mapped."""
    return SECTOR_MAP.get(ticker, "unknown")

