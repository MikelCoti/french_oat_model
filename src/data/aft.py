import requests
import pandas as pd
import numbers
import numpy as np
from io import StringIO, BytesIO
import re
import datetime
from bs4 import BeautifulSoup
from urllib.parse import urljoin
import unicodedata


MONTHS = {
    # English
    "january": 1,
    "february": 2,
    "march": 3,
    "april": 4,
    "may": 5,
    "june": 6,
    "july": 7,
    "august": 8,
    "september": 9,
    "october": 10,
    "november": 11,
    "december": 12,

    # French, after remove_accents()
    "janvier": 1,
    "fevrier": 2,
    "mars": 3,
    "avril": 4,
    "mai": 5,
    "juin": 6,
    "juillet": 7,
    "aout": 8,
    "septembre": 9,
    "octobre": 10,
    "novembre": 11,
    "decembre": 12,
}


SESSION = requests.Session()

SESSION.headers.update({
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/154.0.0.0 Safari/537.36"
    ),
    "Accept": (
        "text/html,application/xhtml+xml,application/xml;"
        "q=0.9,image/avif,image/webp,*/*;q=0.8"
    ),
    "Accept-Language": "en-US,en;q=0.9,fr;q=0.8",
})


URLS = {
    "OAT": "https://www.aft.gouv.fr/en/encours-detaille-oat",
    "OATEI": "https://www.aft.gouv.fr/en/encours-detaille-oatei",
    "OATI": "https://www.aft.gouv.fr/fr/encours-detaille-oati",
    "BTF": "https://www.aft.gouv.fr/fr/encours-detaille-btf",
}

OATI_COEFFICIENT_PAGE = (
    "https://www.aft.gouv.fr/en/oatis-key-figures"
)

OATEI_COEFFICIENT_PAGE = (
    "https://www.aft.gouv.fr/en/oateuroi-key-figures"
)



def remove_accents(text):
    return "".join(
        char
        for char in unicodedata.normalize(
            "NFKD",
            str(text)
        )
        if not unicodedata.combining(
            char
        )
    )


def get_html(url: str) -> str:
    response = SESSION.get(
        url,
        timeout=30
    )

    response.raise_for_status()

    return response.text

def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    rename_map = {}

    for col in df.columns:
        normalized = str(col).strip().lower()

        if "isin" in normalized:
            rename_map[col] = "isin"

        elif (
            "bond" in normalized
            or "libell" in normalized
        ):
            rename_map[col] = "description"

        elif (
            "outstanding" in normalized
            or "encours" in normalized
        ):
            rename_map[col] = "outstanding_eur"

        elif (
            "échéance" in normalized
            or "echeance" in normalized
            or "maturity" in normalized
        ):
            rename_map[col] = "maturity_date"

    return df.rename(columns=rename_map)

def clean_amount(value):
    if pd.isna(value):
        return pd.NA

    text = str(value)

    text = (
        text
        .replace(",", "")
        .replace(" ", "")
        .replace("\xa0", "")
    )

    return pd.to_numeric(
        text,
        errors="coerce"
    )

def parse_coupon(description):

    if pd.isna(description):
        return pd.NA

    text = remove_accents(
        str(description)
    ).lower()

    # AFT uses descriptions such as:
    # "OAT zero coupon 28 March 2028"
    # or potentially the French equivalent "OAT zéro coupon ..."
    if "zero coupon" in text:
        return 0.0

    match = re.search(
        r"([0-9]+(?:[.,][0-9]+)?)\s*%",
        text
    )

    if not match:
        return pd.NA

    return (
        float(
            match.group(1)
            .replace(",", ".")
        )
        / 100
    )

def parse_oat_maturity(description):
    text = str(description)

    match = re.search(
        r"(\d{1,2}\s+[A-Za-z]+\s+\d{4})$",
        text
    )

    if not match:
        return pd.NaT

    return pd.to_datetime(
        match.group(1),
        format="%d %B %Y",
        errors="coerce"
    )


def parse_bond_maturity(
    label
):
    text = (
        remove_accents(label)
        .lower()
    )

    match = re.search(
        r"(\d{1,2})(?:er)?\s+"
        r"([a-z]+)\s+"
        r"(\d{4})",
        text
    )

    if not match:
        return pd.NaT

    day = int(
        match.group(1)
    )

    month_text = (
        match.group(2)
    )

    year = int(
        match.group(3)
    )

    month = MONTHS.get(
        month_text
    )

    if month is None:
        return pd.NaT

    return pd.Timestamp(
        year=year,
        month=month,
        day=day
    )


def extract_security_tables(url):
    html = get_html(url)

    tables = pd.read_html(
        StringIO(html)
    )

    valid_tables = []

    for table in tables:

        table = normalize_columns(table)

        if "isin" not in table.columns:
            continue

        mask = (
            table["isin"]
            .astype(str)
            .str.startswith("FR")
        )

        table = table.loc[mask].copy()

        if not table.empty:
            valid_tables.append(table)

    if not valid_tables:
        raise ValueError(
            f"No securities found at {url}"
        )

    return pd.concat(
        valid_tables,
        ignore_index=True
    )

def scrape_oat_family(
    security_type,
    url
):
    df = extract_security_tables(
        url
    )

    df["security_type"] = (
        security_type
    )

    if "outstanding_eur" in df.columns:

        df["outstanding_eur"] = (
            df["outstanding_eur"]
            .apply(clean_amount)
        )

    # ========================================================
    # BTF
    # ========================================================

    if security_type == "BTF":

        df["coupon_rate"] = 0.0

        # BTF already has its maturity in the
        # separate Échéance column.
        df["maturity_date"] = (
            df["maturity_date"]
            .apply(parse_aft_date)
        )

        df["inflation_index"] = None

    # ========================================================
    # OAT / OATi / OAT€i
    # ========================================================

    else:

        df["coupon_rate"] = (
            df["description"]
            .apply(parse_coupon)
        )

        # Maturity is embedded in description:
        #
        # OAT 2.50% 24 September 2026
        # OATi 0,10 % 1 mars 2028
        #
        df["maturity_date"] = (
            df["description"]
            .apply(parse_aft_date)
        )

        if security_type == "OATEI":

            df["inflation_index"] = (
                "HICPxT"
            )

        elif security_type == "OATI":

            df["inflation_index"] = (
                "FR_CPIxT"
            )

        else:

            df["inflation_index"] = None

    return df

def build_aft_portfolio():

    frames = []

    for security_type, url in URLS.items():

        print(
            f"Scraping {security_type}..."
        )

        df = scrape_oat_family(
            security_type,
            url
        )

        frames.append(df)

    portfolio = pd.concat(
        frames,
        ignore_index=True
    )

    required_columns = [
        "isin",
        "security_type",
        "description",
        "coupon_rate",
        "maturity_date",
        "outstanding_eur",
        "inflation_index",
    ]

    for col in required_columns:
        if col not in portfolio.columns:
            portfolio[col] = pd.NA

    return portfolio[required_columns]


def find_coefficient_file_url(page_url: str) -> str:
    """
    Find the current AFT indexation-coefficient Excel file
    linked from an OATi/OAT€i key-figures page.
    """

    html = get_html(page_url)

    soup = BeautifulSoup(
        html,
        "html.parser"
    )

    candidates = []

    for link in soup.find_all(
        "a",
        href=True
    ):
        href = link["href"]

        text = (
            link.get_text(
                " ",
                strip=True
            )
            .lower()
        )

        href_lower = href.lower()

        is_excel = (
            href_lower.endswith(".xls")
            or href_lower.endswith(".xlsx")
        )

        looks_like_coefficients = (
            "coefficient" in text
            or "coef_oati" in href_lower
            or "coef_oatei" in href_lower
        )

        if (
            is_excel
            and looks_like_coefficients
        ):
            candidates.append(
                urljoin(
                    page_url,
                    href
                )
            )

    if not candidates:
        raise ValueError(
            f"No coefficient Excel file found on {page_url}"
        )

    # Normally there should only be one current file.
    return candidates[0]


def download_excel(
    file_url: str,
    referer: str | None = None
) -> BytesIO:
    headers = {}

    if referer:
        headers["Referer"] = referer

    response = SESSION.get(
        file_url,
        headers=headers,
        timeout=30
    )

    response.raise_for_status()

    return BytesIO(
        response.content
    )


def detect_excel_engine(excel_bytes: BytesIO) -> str:
    """
    Detect whether an Excel file is XLSX or legacy XLS
    from its actual binary contents rather than its filename.
    """

    excel_bytes.seek(0)

    signature = excel_bytes.read(8)

    excel_bytes.seek(0)

    # XLSX files are ZIP archives and therefore start with PK
    if signature.startswith(b"PK"):
        return "openpyxl"

    # Legacy XLS files use the OLE Compound File format
    if signature.startswith(
        b"\xD0\xCF\x11\xE0\xA1\xB1\x1A\xE1"
    ):
        return "xlrd"

    raise ValueError(
        "Downloaded file does not appear to be a valid "
        "XLSX or XLS workbook."
    )



def inspect_coefficient_workbook(
    page_url: str
):
    file_url = find_coefficient_file_url(
        page_url
    )

    excel_bytes = download_excel(
        file_url=file_url,
        referer=page_url
    )

    engine = detect_excel_engine(
        excel_bytes
    )

    print(
        f"Detected Excel engine: {engine}"
    )

    print(
        f"Source URL: {file_url}"
    )

    excel_bytes.seek(0)

    workbook = pd.ExcelFile(
        excel_bytes,
        engine=engine
    )

    print(
        "Sheets:",
        workbook.sheet_names
    )

    for sheet in workbook.sheet_names:

        print(
            f"\n--- {sheet} ---"
        )

        df = pd.read_excel(
            workbook,
            sheet_name=sheet,
            header=None
        )

        print(
            df.head(15)
        )


def find_coefficient_header_row(
    raw: pd.DataFrame
) -> int:
    """
    Find the row containing the coefficient-table headers.
    """

    max_rows = min(
        30,
        len(raw)
    )

    for row_idx in range(
        max_rows
    ):
        row = (
            raw.iloc[row_idx]
            .astype(str)
        )

        row_text = " ".join(
            row.tolist()
        ).lower()

        oat_count = sum(
            "oat" in value.lower()
            for value in row
        )

        has_date = (
            "date" in row_text
            or "jour" in row_text
        )

        if (
            oat_count >= 2
            and has_date
        ):
            return row_idx

    raise ValueError(
        "Could not identify coefficient header row."
    )



def read_coefficient_workbook(
    page_url: str
) -> pd.DataFrame:

    file_url = (
        find_coefficient_file_url(
            page_url
        )
    )

    excel_bytes = download_excel(
        file_url=file_url,
        referer=page_url
    )

    engine = detect_excel_engine(
        excel_bytes
    )

    print(
        f"Reading coefficient file "
        f"with {engine}: {file_url}"
    )

    excel_bytes.seek(0)

    workbook = pd.ExcelFile(
        excel_bytes,
        engine=engine
    )

    all_tables = []

    for sheet in workbook.sheet_names:

        raw = pd.read_excel(
            workbook,
            sheet_name=sheet,
            header=None
        )

        try:
            header_row = (
                find_coefficient_header_row(
                    raw
                )
            )

        except ValueError:
            continue

        excel_bytes.seek(0)

        table = pd.read_excel(
            excel_bytes,
            sheet_name=sheet,
            header=header_row,
            engine=engine
        )

        all_tables.append(
            table
        )

    if not all_tables:
        raise ValueError(
            "No coefficient table found."
        )

    return pd.concat(
        all_tables,
        ignore_index=True
    )


def find_date_column(
    df: pd.DataFrame
):
    for col in df.columns:

        name = (
            str(col)
            .strip()
            .lower()
        )

        if (
            "date" in name
            or "jour" in name
        ):
            return col

    # Fallback: usually first column
    return df.columns[0]


def coefficients_to_long(
    df: pd.DataFrame
) -> pd.DataFrame:

    date_col = find_date_column(
        df
    )

    long_df = df.melt(
        id_vars=[date_col],
        var_name="bond_label",
        value_name="index_coefficient"
    )

    long_df = long_df.rename(
        columns={
            date_col: "date"
        }
    )

    long_df["date"] = pd.to_datetime(
        long_df["date"],
        errors="coerce"
    )

    long_df["index_coefficient"] = (
        pd.to_numeric(
            long_df[
                "index_coefficient"
            ],
            errors="coerce"
        )
    )

    long_df = long_df.dropna(
        subset=[
            "date",
            "index_coefficient"
        ]
    )

    return (
        long_df
        .sort_values(
            ["bond_label", "date"]
        )
        .reset_index(drop=True)
    )


def latest_coefficients(
    coeff_long: pd.DataFrame,
    snapshot_date
) -> pd.DataFrame:

    snapshot_date = pd.Timestamp(
        snapshot_date
    )

    available = coeff_long[
        coeff_long["date"]
        <= snapshot_date
    ].copy()

    latest = (
        available
        .sort_values("date")
        .groupby(
            "bond_label",
            as_index=False
        )
        .tail(1)
    )

    return latest.reset_index(
        drop=True
    )


def match_coefficients_to_portfolio(
    latest_coefficients_df,
    portfolio,
    security_type
):
    coeff = (
        latest_coefficients_df
        .copy()
    )

    coeff["coupon_key"] = (
        coeff["coupon_rate"]
        .round(6)
    )

    indexed = portfolio[
        portfolio[
            "security_type"
        ] == security_type
    ].copy()

    indexed["coupon_key"] = (
        indexed["coupon_rate"]
        .round(6)
    )

    matched = coeff.merge(
        indexed[
            [
                "isin",
                "coupon_key",
                "maturity_date"
            ]
        ],
        on=[
            "coupon_key",
            "maturity_date"
        ],
        how="left"
    )

    return matched[
        [
            "isin",
            "date",
            "index_coefficient",
            "bond_label"
        ]
    ]


def get_current_index_coefficients(
    portfolio,
    snapshot_date
):
    frames = []

    sources = {
        "OATEI": (
            OATEI_COEFFICIENT_PAGE
        ),
        "OATI": (
            OATI_COEFFICIENT_PAGE
        ),
    }

    for security_type, page in (
        sources.items()
    ):
        wide = (
            read_coefficient_workbook(
                page
            )
        )

        long = coefficients_to_long(
            wide
        )

        latest = latest_coefficients(
            long,
            snapshot_date
        )

        latest[
            "coupon_rate"
        ] = (
            latest["bond_label"]
            .apply(parse_coupon)
        )

        latest[
            "maturity_date"
        ] = (
            latest["bond_label"]
            .apply(
                parse_bond_maturity
            )
        )

        matched = (
            match_coefficients_to_portfolio(
                latest,
                portfolio,
                security_type
            )
        )

        matched[
            "security_type"
        ] = security_type

        frames.append(
            matched
        )

    result = pd.concat(
        frames,
        ignore_index=True
    )

    return result


def parse_percentage(value):
    """
    Convert an Excel percentage or textual percentage to decimal form.

    Examples:
        0.001     -> 0.001
        "0,10%"   -> 0.001
        "1.85%"   -> 0.0185
    """

    if pd.isna(value):
        return np.nan

    if isinstance(value, numbers.Number):
        return float(value)

    text = str(value).strip()

    if not text:
        return np.nan

    text = text.replace(",", ".")

    if "%" in text:
        text = text.replace("%", "")
        return float(text) / 100

    return float(text)


def parse_excel_date(value):
    """
    Parse either an Excel datetime or a dd/mm/yyyy-style value.
    """

    if pd.isna(value):
        return pd.NaT

    if isinstance(
        value,
        (pd.Timestamp,)
    ):
        return value.normalize()

    # Excel serial date fallback
    if isinstance(value, numbers.Number):
        if 20_000 < value < 80_000:
            return (
                pd.Timestamp("1899-12-30")
                + pd.to_timedelta(
                    value,
                    unit="D"
                )
            ).normalize()

    return pd.to_datetime(
        value,
        dayfirst=True,
        errors="coerce"
    )


def find_bond_type_row(raw):
    """
    Find the row containing the repeated OATi/OAT€i labels.
    """

    best_row = None
    best_count = 0

    for row_idx in range(
        min(10, len(raw))
    ):

        count = sum(
            "OAT" in str(value).upper()
            for value in raw.iloc[row_idx]
            if pd.notna(value)
        )

        if count > best_count:
            best_count = count
            best_row = row_idx

    if (
        best_row is None
        or best_count == 0
    ):
        raise ValueError(
            "Could not find OAT bond metadata row."
        )

    return best_row


def extract_bond_metadata(raw):
    type_row = find_bond_type_row(
        raw
    )

    coupon_row = type_row + 1
    maturity_row = type_row + 2

    records = []

    # Skip columns A and B
    for col in raw.columns[2:]:

        security_type = raw.iloc[
            type_row,
            col
        ]

        if (
            pd.isna(security_type)
            or "OAT" not in str(
                security_type
            ).upper()
        ):
            continue

        coupon = parse_percentage(
            raw.iloc[
                coupon_row,
                col
            ]
        )

        maturity = parse_excel_date(
            raw.iloc[
                maturity_row,
                col
            ]
        )

        if pd.isna(maturity):
            continue

        records.append({
            "excel_column": col,
            "security_type": (
                str(security_type)
                .strip()
            ),
            "coupon_rate": coupon,
            "maturity_date": maturity
        })

    return pd.DataFrame(
        records
    )


def find_daily_data_start(
    raw,
    minimum_row=8
):
    """
    Find the first row whose first column contains a valid date.
    """

    for row_idx in range(
        minimum_row,
        len(raw)
    ):

        date = parse_excel_date(
            raw.iloc[
                row_idx,
                0
            ]
        )

        if pd.notna(date):
            return row_idx

    raise ValueError(
        "Could not find start of daily coefficient data."
    )


def parse_coefficient_timeseries(
    raw,
    metadata
):
    data_start = find_daily_data_start(
        raw
    )

    records = []

    for row_idx in range(
        data_start,
        len(raw)
    ):

        date = parse_excel_date(
            raw.iloc[
                row_idx,
                0
            ]
        )

        if pd.isna(date):
            continue

        daily_inflation_reference = (
            pd.to_numeric(
                str(
                    raw.iloc[
                        row_idx,
                        1
                    ]
                ).replace(
                    ",",
                    "."
                ),
                errors="coerce"
            )
        )

        for _, bond in (
            metadata.iterrows()
        ):

            col = int(
                bond["excel_column"]
            )

            raw_coefficient = (
                raw.iloc[
                    row_idx,
                    col
                ]
            )

            coefficient = pd.to_numeric(
                str(
                    raw_coefficient
                ).replace(
                    ",",
                    "."
                ),
                errors="coerce"
            )

            # Blank cells occur before issuance or after maturity.
            if pd.isna(
                coefficient
            ):
                continue

            records.append({
                "date": date,

                "daily_inflation_reference": (
                    daily_inflation_reference
                ),

                "security_type": (
                    bond[
                        "security_type"
                    ]
                ),

                "coupon_rate": (
                    bond[
                        "coupon_rate"
                    ]
                ),

                "maturity_date": (
                    bond[
                        "maturity_date"
                    ]
                ),

                "index_coefficient": (
                    float(
                        coefficient
                    )
                )
            })

    return pd.DataFrame(
        records
    )


def coefficients_at_date(
    coefficients,
    snapshot_date
):
    snapshot_date = pd.Timestamp(
        snapshot_date
    )

    available = coefficients[
        coefficients["date"]
        <= snapshot_date
    ].copy()

    latest = (
        available
        .sort_values("date")
        .groupby(
            [
                "coupon_rate",
                "maturity_date"
            ],
            as_index=False
        )
        .tail(1)
        .reset_index(drop=True)
    )

    return latest


def add_matching_keys(
    df: pd.DataFrame
) -> pd.DataFrame:

    df = df.copy()

    required = {
        "coupon_rate",
        "maturity_date",
    }

    missing = (
        required
        - set(df.columns)
    )

    if missing:
        raise ValueError(
            "Cannot create matching keys. "
            f"Missing columns: {missing}"
        )

    df["coupon_key"] = (
        pd.to_numeric(
            df["coupon_rate"],
            errors="coerce"
        )
        .round(8)
    )

    df["maturity_key"] = (
        df["maturity_date"]
        .apply(parse_aft_date)
    )

    return df


def read_aft_coefficient_file(
    page_url: str
) -> pd.DataFrame:

    file_url = (
        find_coefficient_file_url(
            page_url
        )
    )

    excel_bytes = download_excel(
        file_url=file_url,
        referer=page_url
    )

    engine = detect_excel_engine(
        excel_bytes
    )

    excel_bytes.seek(0)

    raw = pd.read_excel(
        excel_bytes,
        sheet_name=0,
        header=None,
        engine=engine
    )

    metadata = (
        extract_bond_metadata(
            raw
        )
    )

    coefficients = (
        parse_coefficient_timeseries(
            raw,
            metadata
        )
    )

    return coefficients



def parse_aft_date(value):
    """
    Parse dates used by AFT.

    Handles:
    - pandas/Excel dates
    - Excel serial numbers
    - dd/mm/yyyy
    - dd-mm-yyyy
    - yyyy-mm-dd
    - English month names
    - French month names
    - dates embedded inside bond descriptions

    Examples:
        "OATi 0,10 % 1 mars 2028"
        "OATi 3,40 % 25 juillet 2029"
        "07 octobre 2026"
        "25 November 2026"
    """

    if pd.isna(value):
        return pd.NaT

    # --------------------------------------------------------
    # Already a datetime
    # --------------------------------------------------------

    if isinstance(
        value,
        (pd.Timestamp, datetime.datetime, datetime.date)
    ):
        return pd.Timestamp(value).normalize()

    # --------------------------------------------------------
    # Excel serial date
    # --------------------------------------------------------

    if isinstance(value, numbers.Number):

        if 20_000 < float(value) < 80_000:

            return (
                pd.Timestamp("1899-12-30")
                + pd.to_timedelta(
                    float(value),
                    unit="D"
                )
            ).normalize()

    # --------------------------------------------------------
    # Normalize text
    # --------------------------------------------------------

    text = remove_accents(
        str(value)
    )

    text = (
        text
        .replace("\xa0", " ")
        .replace("\u202f", " ")
        .strip()
        .lower()
    )

    # Collapse multiple spaces
    text = re.sub(
        r"\s+",
        " ",
        text
    )

    # --------------------------------------------------------
    # Explicit numeric date formats
    # --------------------------------------------------------

    numeric_match = re.search(
        r"\b"
        r"(\d{1,2})"
        r"[/-]"
        r"(\d{1,2})"
        r"[/-]"
        r"(\d{4})"
        r"\b",
        text
    )

    if numeric_match:

        day = int(
            numeric_match.group(1)
        )

        month = int(
            numeric_match.group(2)
        )

        year = int(
            numeric_match.group(3)
        )

        try:

            return pd.Timestamp(
                year=year,
                month=month,
                day=day
            )

        except ValueError:

            return pd.NaT

    # --------------------------------------------------------
    # ISO yyyy-mm-dd
    # --------------------------------------------------------

    iso_match = re.search(
        r"\b"
        r"(\d{4})"
        r"-"
        r"(\d{1,2})"
        r"-"
        r"(\d{1,2})"
        r"\b",
        text
    )

    if iso_match:

        year = int(
            iso_match.group(1)
        )

        month = int(
            iso_match.group(2)
        )

        day = int(
            iso_match.group(3)
        )

        try:

            return pd.Timestamp(
                year=year,
                month=month,
                day=day
            )

        except ValueError:

            return pd.NaT

    # --------------------------------------------------------
    # Named month:
    #
    # "1 mars 2028"
    # "25 juillet 2029"
    # "07 octobre 2026"
    # "25 November 2026"
    # --------------------------------------------------------

    named_match = re.search(
        r"\b"
        r"(\d{1,2})"
        r"(?:er)?"
        r"\s+"
        r"([a-z]+)"
        r"\s+"
        r"(\d{4})"
        r"\b",
        text
    )

    if named_match:

        day = int(
            named_match.group(1)
        )

        month_name = (
            named_match.group(2)
        )

        year = int(
            named_match.group(3)
        )

        month = MONTHS.get(
            month_name
        )

        if month is None:

            return pd.NaT

        try:

            return pd.Timestamp(
                year=year,
                month=month,
                day=day
            )

        except ValueError:

            return pd.NaT

    return pd.NaT


def test_aft_date_parser():

    cases = {
        "OATi 0,10 % 1 mars 2028":
            "2028-03-01",

        "OATi 3,40 % 25 juillet 2029":
            "2029-07-25",

        "OATi 0,10 % 1 mars 2032":
            "2032-03-01",

        "OATi 0,10 % 1 mars 2036":
            "2036-03-01",

        "OATi 0,55 % 1 mars 2039":
            "2039-03-01",

        "07 octobre 2026":
            "2026-10-07",

        "14 octobre 2026":
            "2026-10-14",

        "04 novembre 2026":
            "2026-11-04",

        "02 décembre 2026":
            "2026-12-02",

        "OAT 2.50% 24 September 2026":
            "2026-09-24",
    }

    for raw, expected in cases.items():

        actual = parse_aft_date(
            raw
        )

        expected = pd.Timestamp(
            expected
        )

        assert actual == expected, (
            f"\nFailed to parse:\n"
            f"  raw:      {raw!r}\n"
            f"  expected: {expected}\n"
            f"  actual:   {actual}"
        )

    print(
        "AFT date parser: all tests passed."
    )


def test_coupon_parser():

    cases = {
        "OAT 2.50% 24 September 2026":
            0.025,

        "OATi 0,10 % 1 mars 2028":
            0.001,

        "OAT€i 1.85% 25 July 2027":
            0.0185,

        "OAT zero coupon 28 March 2028":
            0.0,

        "OAT zéro coupon 28 mars 2028":
            0.0,
    }

    for raw, expected in cases.items():

        actual = parse_coupon(
            raw
        )

        assert abs(
            actual - expected
        ) < 1e-12, (
            f"\nFailed coupon parse:\n"
            f"  raw:      {raw!r}\n"
            f"  expected: {expected}\n"
            f"  actual:   {actual}"
        )

    print(
        "AFT coupon parser: all tests passed."
    )


if __name__ == "__main__":

    from pathlib import Path

    # ========================================================
    # CONFIGURATION
    # ========================================================

    SNAPSHOT_DATE = pd.Timestamp("2026-10-03")

    OUTPUT_PATH = Path(
        "data/processed/"
        f"aft_portfolio_{SNAPSHOT_DATE.date()}.csv"
    )

    print("=" * 78)
    print("AFT END-TO-END PORTFOLIO VALIDATION")
    print("=" * 78)

    print(
        "\n[0] Testing AFT parsers..."
    )

    test_aft_date_parser()
    test_coupon_parser()

    # ========================================================
    # 1. SCRAPE CURRENT AFT PORTFOLIO
    # ========================================================

    print("\n[1] Scraping AFT securities...")

    portfolio = build_aft_portfolio()

    assert not portfolio.empty, (
        "Portfolio scraper returned no securities."
    )

    print(
        f"Raw securities scraped: "
        f"{len(portfolio):,}"
    )

    # --------------------------------------------------------
    # Standardize basic fields
    # --------------------------------------------------------

    portfolio["maturity_date"] = (
        portfolio["maturity_date"]
        .apply(parse_aft_date)
    )

    portfolio["coupon_rate"] = pd.to_numeric(
        portfolio["coupon_rate"],
        errors="coerce"
    )

    portfolio["outstanding_eur"] = pd.to_numeric(
        portfolio["outstanding_eur"],
        errors="coerce"
    )

    portfolio["security_type"] = (
        portfolio["security_type"]
        .astype(str)
        .str.strip()
        .str.upper()
        .replace({
            "OAT€I": "OATEI",
            "OATÉI": "OATEI",
        })
    )

    # ========================================================
    # 2. BASIC PORTFOLIO VALIDATION
    # ========================================================

    print("\n[2] Validating scraped portfolio...")

    required_columns = {
        "isin",
        "security_type",
        "coupon_rate",
        "maturity_date",
        "outstanding_eur",
    }

    missing_columns = (
        required_columns
        - set(portfolio.columns)
    )

    assert not missing_columns, (
        "Portfolio is missing required columns: "
        f"{missing_columns}"
    )

    # ISIN must exist
    assert not portfolio[
        "isin"
    ].isna().any(), (
        "NULL ISIN found."
    )

    # ISIN should uniquely identify each security
    duplicate_isins = portfolio[
        portfolio["isin"].duplicated(
            keep=False
        )
    ]

    assert duplicate_isins.empty, (
        "Duplicate ISINs found:\n"
        f"{duplicate_isins.to_string(index=False)}"
    )

    # Testing this thing
    bad_maturities = portfolio[
        portfolio["maturity_date"].isna()
    ].copy()

    if not bad_maturities.empty:
        print(
            "\nSecurities with invalid maturity dates:"
        )

        print(
            bad_maturities[
                [
                    "isin",
                    "security_type",
                    "description",
                ]
            ].to_string(
                index=False
            )
        )

    # Maturity must exist
    assert not portfolio[
        "maturity_date"
    ].isna().any(), (
        "Invalid maturity dates found."
    )

    # Principal must be valid
    assert not portfolio[
        "outstanding_eur"
    ].isna().any(), (
        "Invalid outstanding amounts found."
    )

    assert (
        portfolio[
            "outstanding_eur"
        ] >= 0
    ).all(), (
        "Non-positive outstanding principal found."
    )

    print(
        "Basic portfolio structure: OK"
    )

    bad_coupons = portfolio[
        portfolio["coupon_rate"].isna()
    ].copy()

    if not bad_coupons.empty:

        print(
            "\nSecurities with invalid coupons:"
        )

        print(
            bad_coupons[
                [
                    "isin",
                    "security_type",
                    "description",
                ]
            ].to_string(
                index=False
            )
        )

    assert bad_coupons.empty, (
        "At least one security has an invalid coupon."
    )

    # ========================================================
    # 3. REMOVE SECURITIES ALREADY MATURED
    # ========================================================

    print(
        f"\n[3] Applying snapshot date "
        f"{SNAPSHOT_DATE.date()}..."
    )

    matured = portfolio[
        portfolio["maturity_date"]
        < SNAPSHOT_DATE
    ].copy()

    if not matured.empty:

        print(
            f"Removing {len(matured)} "
            "already-matured securities:"
        )

        print(
            matured[
                [
                    "isin",
                    "security_type",
                    "description",
                    "maturity_date",
                ]
            ].to_string(
                index=False
            )
        )

    portfolio = portfolio[
        portfolio["maturity_date"]
        >= SNAPSHOT_DATE
    ].copy()

    portfolio = portfolio.reset_index(
        drop=True
    )

    print(
        f"Live securities remaining: "
        f"{len(portfolio):,}"
    )

    # ========================================================
    # 4. SECURITY-TYPE SANITY CHECK
    # ========================================================

    print("\n[4] Security breakdown...")

    valid_types = {
        "OAT",
        "OATI",
        "OATEI",
        "BTF",
    }

    unknown_types = (
        set(
            portfolio[
                "security_type"
            ].unique()
        )
        - valid_types
    )

    assert not unknown_types, (
        "Unexpected security types found: "
        f"{unknown_types}"
    )

    breakdown = (
        portfolio
        .groupby("security_type")
        .agg(
            securities=(
                "isin",
                "count"
            ),
            face_value_eur=(
                "outstanding_eur",
                "sum"
            )
        )
    )

    breakdown[
        "face_value_bn"
    ] = (
        breakdown[
            "face_value_eur"
        ]
        / 1e9
    )

    print(
        breakdown[
            [
                "securities",
                "face_value_bn",
            ]
        ]
        .round(2)
        .to_string()
    )

    # We expect all four categories
    for expected_type in valid_types:

        assert expected_type in set(
            portfolio[
                "security_type"
            ]
        ), (
            f"No {expected_type} "
            "securities found."
        )

    # ========================================================
    # 5. READ OAT€i COEFFICIENT HISTORY
    # ========================================================

    print(
        "\n[5] Reading OAT€i "
        "coefficient workbook..."
    )

    oatei_history = (
        read_aft_coefficient_file(
            OATEI_COEFFICIENT_PAGE
        )
    )

    assert not oatei_history.empty

    print(
        f"OAT€i coefficient observations: "
        f"{len(oatei_history):,}"
    )

    print(
        f"Range: "
        f"{oatei_history['date'].min().date()} "
        f"-> "
        f"{oatei_history['date'].max().date()}"
    )

    # ========================================================
    # 6. READ OATi COEFFICIENT HISTORY
    # ========================================================

    print(
        "\n[6] Reading OATi "
        "coefficient workbook..."
    )

    oati_history = (
        read_aft_coefficient_file(
            OATI_COEFFICIENT_PAGE
        )
    )

    assert not oati_history.empty

    print(
        f"OATi coefficient observations: "
        f"{len(oati_history):,}"
    )

    print(
        f"Range: "
        f"{oati_history['date'].min().date()} "
        f"-> "
        f"{oati_history['date'].max().date()}"
    )

    # ========================================================
    # 7. SELECT COEFFICIENTS AT SNAPSHOT DATE
    # ========================================================

    print(
        "\n[7] Selecting coefficients "
        "at snapshot date..."
    )

    oatei_latest = coefficients_at_date(
        oatei_history,
        SNAPSHOT_DATE
    )

    oati_latest = coefficients_at_date(
        oati_history,
        SNAPSHOT_DATE
    )

    assert not oatei_latest.empty
    assert not oati_latest.empty

    assert (
        oatei_latest["date"]
        <= SNAPSHOT_DATE
    ).all()

    assert (
        oati_latest["date"]
        <= SNAPSHOT_DATE
    ).all()

    print(
        f"OAT€i bonds with coefficients: "
        f"{len(oatei_latest)}"
    )

    print(
        f"OATi bonds with coefficients: "
        f"{len(oati_latest)}"
    )

    # ========================================================
    # 8. CREATE MATCHING KEYS
    # ========================================================

    print(
        "\n[8] Preparing linker matching keys..."
    )

    portfolio = add_matching_keys(
        portfolio
    )

    oatei_latest = add_matching_keys(
        oatei_latest
    )

    oati_latest = add_matching_keys(
        oati_latest
    )

    # Validate immediately instead of failing later
    for name, df in {
        "Portfolio": portfolio,
        "OAT€i": oatei_latest,
        "OATi": oati_latest,
    }.items():

        required_keys = {
            "coupon_key",
            "maturity_key",
        }

        missing_keys = (
            required_keys
            - set(df.columns)
        )

        assert not missing_keys, (
            f"{name}: matching keys were not created: "
            f"{missing_keys}"
        )

        assert not df[
            "coupon_key"
        ].isna().any(), (
            f"{name}: NULL coupon_key found."
        )

        assert not df[
            "maturity_key"
        ].isna().any(), (
            f"{name}: NULL maturity_key found."
        )

        print(
            f"{name}: matching keys OK"
        )

    print(
        "\nOAT€i columns:",
        oatei_latest.columns.tolist()
    )

    print(
        "OATi columns:",
        oati_latest.columns.tolist()
    )


    # ========================================================
    # 9. VALIDATE COEFFICIENT TABLE UNIQUENESS
    # ========================================================

    print(
        "\n[9] Checking coefficient-key "
        "uniqueness..."
    )

    coefficient_key = [
        "coupon_key",
        "maturity_key",
    ]

    for name, df in {
        "OAT€i": oatei_latest,
        "OATi": oati_latest,
    }.items():

        duplicates = df[
            df.duplicated(
                subset=coefficient_key,
                keep=False
            )
        ]

        assert duplicates.empty, (
            f"{name}: duplicate coefficient "
            "keys found:\n"
            f"{duplicates.to_string(index=False)}"
        )

        print(
            f"{name}: unique bond keys OK"
        )

    # ========================================================
    # 10. MATCH OAT€i TO PORTFOLIO
    # ========================================================

    print(
        "\n[10] Matching OAT€i securities..."
    )

    live_oatei = portfolio[
        portfolio[
            "security_type"
        ] == "OATEI"
    ].copy()

    oatei_matches = (
        live_oatei[
            [
                "isin",
                "coupon_key",
                "maturity_key",
            ]
        ]
        .merge(
            oatei_latest[
                [
                    "coupon_key",
                    "maturity_key",
                    "date",
                    "index_coefficient",
                ]
            ],
            on=[
                "coupon_key",
                "maturity_key",
            ],
            how="left",
            validate="one_to_one"
        )
        .rename(
            columns={
                "date":
                    "coefficient_date"
            }
        )
    )

    unmatched_oatei = oatei_matches[
        oatei_matches[
            "index_coefficient"
        ].isna()
    ]

    if not unmatched_oatei.empty:

        print(
            "\nUNMATCHED OAT€i:"
        )

        print(
            unmatched_oatei.to_string(
                index=False
            )
        )

    assert unmatched_oatei.empty, (
        f"{len(unmatched_oatei)} "
        "live OAT€i securities could "
        "not be matched."
    )

    print(
        f"Matched OAT€i: "
        f"{len(oatei_matches)} / "
        f"{len(live_oatei)}"
    )

    # ========================================================
    # 11. MATCH OATi TO PORTFOLIO
    # ========================================================

    print(
        "\n[11] Matching OATi securities..."
    )

    live_oati = portfolio[
        portfolio[
            "security_type"
        ] == "OATI"
    ].copy()

    oati_matches = (
        live_oati[
            [
                "isin",
                "coupon_key",
                "maturity_key",
            ]
        ]
        .merge(
            oati_latest[
                [
                    "coupon_key",
                    "maturity_key",
                    "date",
                    "index_coefficient",
                ]
            ],
            on=[
                "coupon_key",
                "maturity_key",
            ],
            how="left",
            validate="one_to_one"
        )
        .rename(
            columns={
                "date":
                    "coefficient_date"
            }
        )
    )

    unmatched_oati = oati_matches[
        oati_matches[
            "index_coefficient"
        ].isna()
    ]

    if not unmatched_oati.empty:

        print(
            "\nUNMATCHED OATi:"
        )

        print(
            unmatched_oati.to_string(
                index=False
            )
        )

    assert unmatched_oati.empty, (
        f"{len(unmatched_oati)} "
        "live OATi securities could "
        "not be matched."
    )

    print(
        f"Matched OATi: "
        f"{len(oati_matches)} / "
        f"{len(live_oati)}"
    )

    # ========================================================
    # 12. MERGE COEFFICIENTS INTO FINAL PORTFOLIO
    # ========================================================

    print(
        "\n[12] Building final portfolio..."
    )

    coefficient_map = pd.concat(
        [
            oatei_matches[
                [
                    "isin",
                    "coefficient_date",
                    "index_coefficient",
                ]
            ],
            oati_matches[
                [
                    "isin",
                    "coefficient_date",
                    "index_coefficient",
                ]
            ],
        ],
        ignore_index=True
    )

    # Remove any old coefficient columns
    # before performing the definitive merge.

    for col in [
        "index_coefficient",
        "coefficient_date",
    ]:

        if col in portfolio.columns:
            portfolio = portfolio.drop(
                columns=col
            )

    portfolio = portfolio.merge(
        coefficient_map,
        on="isin",
        how="left",
        validate="one_to_one"
    )

    # Nominal debt has no inflation
    # indexation coefficient.

    nominal_mask = portfolio[
        "security_type"
    ].isin(
        [
            "OAT",
            "BTF",
        ]
    )

    portfolio.loc[
        nominal_mask,
        "index_coefficient"
    ] = 1.0

    # We can use the snapshot date for
    # nominal securities as a harmless
    # bookkeeping value.

    portfolio.loc[
        nominal_mask,
        "coefficient_date"
    ] = SNAPSHOT_DATE

    # Ensure inflation-index labels exist.

    portfolio.loc[
        portfolio[
            "security_type"
        ] == "OATEI",
        "inflation_index"
    ] = "HICPxT"

    portfolio.loc[
        portfolio[
            "security_type"
        ] == "OATI",
        "inflation_index"
    ] = "FR_CPIxT"

    # ========================================================
    # 13. FINAL INDEXATION VALIDATION
    # ========================================================

    print(
        "\n[13] Validating indexation data..."
    )

    indexed_mask = portfolio[
        "security_type"
    ].isin(
        [
            "OATI",
            "OATEI",
        ]
    )

    assert not portfolio.loc[
        indexed_mask,
        "index_coefficient"
    ].isna().any(), (
        "Some indexed securities still "
        "have no coefficient."
    )

    assert (
        portfolio.loc[
            indexed_mask,
            "index_coefficient"
        ] > 0
    ).all(), (
        "Non-positive index coefficient found."
    )

    assert (
        portfolio.loc[
            indexed_mask,
            "coefficient_date"
        ]
        <= SNAPSHOT_DATE
    ).all(), (
        "Coefficient from after snapshot "
        "date was selected."
    )

    print(
        "All live OATi/OAT€i have "
        "valid coefficients."
    )

    # ========================================================
    # 14. CALCULATE INDEXED PRINCIPAL
    # ========================================================

    print(
        "\n[14] Calculating indexed principal..."
    )

    portfolio[
        "indexed_outstanding_eur"
    ] = portfolio[
        "outstanding_eur"
    ]

    portfolio.loc[
        indexed_mask,
        "indexed_outstanding_eur"
    ] = (
        portfolio.loc[
            indexed_mask,
            "outstanding_eur"
        ]
        *
        portfolio.loc[
            indexed_mask,
            "index_coefficient"
        ]
    )

    assert (
        portfolio[
            "indexed_outstanding_eur"
        ] >= 0
    ).all()

    # ========================================================
    # 15. PORTFOLIO-LEVEL SANITY CHECKS
    # ========================================================

    print(
        "\n[15] Running portfolio-level "
        "sanity checks..."
    )

    face_total = portfolio[
        "outstanding_eur"
    ].sum()

    indexed_total = portfolio[
        "indexed_outstanding_eur"
    ].sum()

    face_total_trn = (
        face_total / 1e12
    )

    indexed_total_trn = (
        indexed_total / 1e12
    )

    print(
        f"Face-value portfolio: "
        f"€{face_total_trn:.3f}tn"
    )

    print(
        f"Indexed liability portfolio: "
        f"€{indexed_total_trn:.3f}tn"
    )

    # Very broad sanity bounds. These are intended
    # to catch parser/unit failures, not enforce
    # a particular official debt-stock estimate.

    assert (
        1.5e12
        < face_total
        < 4.5e12
    ), (
        "Portfolio total is implausible. "
        "Possible unit/parsing error."
    )

    assert (
        indexed_total
        >= face_total
        * 0.95
    ), (
        "Indexed debt total looks implausibly "
        "low relative to face value."
    )

    # ========================================================
    # 16. SHOW LINKERS FOR MANUAL SPOT CHECK
    # ========================================================

    print(
        "\n[16] Final indexed securities:"
    )

    linker_display = (
        portfolio.loc[
            indexed_mask,
            [
                "isin",
                "security_type",
                "coupon_rate",
                "maturity_date",
                "outstanding_eur",
                "coefficient_date",
                "index_coefficient",
                "indexed_outstanding_eur",
            ]
        ]
        .sort_values(
            [
                "security_type",
                "maturity_date",
            ]
        )
    )

    print(
        linker_display.to_string(
            index=False
        )
    )

    # ========================================================
    # 17. CLEAN TEMPORARY MATCHING COLUMNS
    # ========================================================

    portfolio = portfolio.drop(
        columns=[
            "coupon_key",
            "maturity_key",
        ],
        errors="ignore"
    )

    # ========================================================
    # 18. SAVE FINAL DATASET
    # ========================================================

    print(
        "\n[17] Saving validated portfolio..."
    )

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    portfolio.to_csv(
        OUTPUT_PATH,
        index=False
    )

    print(
        f"Saved:\n{OUTPUT_PATH.resolve()}"
    )

    # ========================================================
    # FINAL SUCCESS MESSAGE
    # ========================================================

    print("\n" + "=" * 78)
    print(
        "ALL AFT END-TO-END TESTS PASSED"
    )
    print("=" * 78)

    print(
        "\nThe generated portfolio is ready "
        "for debt_portfolio.py."
    )