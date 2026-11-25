import streamlit as st
import pandas as pd
import hashlib
import re
import subprocess
import tempfile
import shutil
import zipfile

from pathlib import Path
from collections import defaultdict, Counter
from docx import Document
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from io import BytesIO

try:
    import fitz
except ImportError:
    fitz = None

try:
    import pdfplumber
except ImportError:
    pdfplumber = None

try:
    import pytesseract
    from PIL import Image
    OCR_AVAILABLE = True
except ImportError:
    pytesseract = None
    Image = None
    OCR_AVAILABLE = False


# ============================================================
# CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Resume Duplicate Tracking System",
    page_icon="📄",
    layout="wide"
)

BASE_DIR = Path(__file__).resolve().parent

# Use a project-relative folder so the app works on Streamlit Community Cloud
# as well as on Windows. For online testing, upload files through the uploader.
RESUME_FOLDER = BASE_DIR / "resumes"

OUTPUT_FOLDER = BASE_DIR / "output"

SUPPORTED_FILES = {
    ".pdf", ".doc", ".docx", ".xls", ".xlsx", ".csv", ".txt"
}

SIMILARITY_THRESHOLD = 0.80

GENERIC_FOLDERS = {
    "resume", "resumes", "candidate", "candidates",
    "uploads", "upload", "documents", "files",
    "data", "applications"
}


# ============================================================
# FOLDER SETUP AND NORMALIZATION
# ============================================================

def create_folders():
    RESUME_FOLDER.mkdir(parents=True, exist_ok=True)
    OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)


def clean_cell(value):
    if value is None:
        return ""

    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass

    return str(value).strip()


def normalize_candidate_id(value):
    return re.sub(r"\s+", "", clean_cell(value)).upper()


def normalize_email(value):
    value = clean_cell(value).lower()

    match = re.search(
        r"[a-z0-9._%+-]+@[a-z0-9.-]+\.[a-z]{2,}",
        value,
        re.I
    )

    return match.group(0).lower() if match else ""


def normalize_phone(value):
    value = clean_cell(value)

    if re.fullmatch(r"\d+\.0", value):
        value = value[:-2]

    digits = re.sub(r"\D", "", value)

    if len(digits) >= 10:
        return digits[-10:]

    return digits if len(digits) >= 7 else ""


def normalize_text(value):
    return re.sub(
        r"\s+", " ", clean_cell(value).lower()
    ).strip()


def get_file_hash(file_path):
    sha = hashlib.sha256()

    try:
        with open(file_path, "rb") as file:
            for chunk in iter(lambda: file.read(8192), b""):
                sha.update(chunk)

        return sha.hexdigest()

    except Exception:
        return ""


# ============================================================
# TEXT EXTRACTION
# ============================================================

def extract_pdf_text(file_path, enable_ocr=True):
    pages = []

    if fitz is not None:
        try:
            with fitz.open(file_path) as pdf:

                for page in pdf:
                    page_text = page.get_text("text") or ""

                    if (
                        not page_text.strip()
                        and enable_ocr
                        and OCR_AVAILABLE
                    ):
                        try:
                            pix = page.get_pixmap(
                                matrix=fitz.Matrix(2, 2),
                                alpha=False
                            )

                            image = Image.open(
                                BytesIO(pix.tobytes("png"))
                            )

                            page_text = pytesseract.image_to_string(
                                image
                            )

                        except Exception:
                            pass

                    pages.append(page_text)

            result = "\n".join(pages)

            if result.strip():
                return result

        except Exception:
            pass

    if pdfplumber is not None:
        try:
            with pdfplumber.open(file_path) as pdf:
                return "\n".join(
                    page.extract_text() or ""
                    for page in pdf.pages
                )

        except Exception:
            pass

    return ""


def extract_docx_text(file_path):
    document = Document(file_path)

    content = [
        p.text for p in document.paragraphs
        if p.text.strip()
    ]

    for table in document.tables:
        for row in table.rows:
            content.append(
                " ".join(cell.text for cell in row.cells)
            )

    return "\n".join(content)


def extract_doc_text(file_path):
    libreoffice = (
        shutil.which("libreoffice")
        or shutil.which("soffice")
    )

    if not libreoffice:
        raise RuntimeError(
            "LibreOffice is required for DOC files."
        )

    with tempfile.TemporaryDirectory() as temp_dir:

        result = subprocess.run(
            [
                libreoffice,
                "--headless",
                "--convert-to",
                "txt:Text",
                "--outdir",
                temp_dir,
                str(file_path)
            ],
            capture_output=True,
            text=True,
            timeout=90
        )

        output_file = (
            Path(temp_dir) /
            (Path(file_path).stem + ".txt")
        )

        if result.returncode != 0 or not output_file.exists():
            raise RuntimeError(
                result.stderr or "DOC conversion failed."
            )

        return output_file.read_text(
            encoding="utf-8",
            errors="ignore"
        )


def extract_text(file_path, enable_ocr=True):
    extension = Path(file_path).suffix.lower()

    try:
        if extension == ".pdf":
            text = extract_pdf_text(file_path, enable_ocr)

        elif extension == ".docx":
            text = extract_docx_text(file_path)

        elif extension == ".doc":
            text = extract_doc_text(file_path)

        elif extension == ".txt":
            text = Path(file_path).read_text(
                encoding="utf-8",
                errors="ignore"
            )

        else:
            return "", "Unsupported document format."

        if not text.strip():
            return "", "No readable text found."

        return text, ""

    except Exception as error:
        return "", str(error)


# ============================================================
# CANDIDATE DETAILS
# ============================================================

def extract_candidate_id(text):
    patterns = [
        r"(?:Candidate\s*ID|Applicant\s*ID|Registration\s*ID|Student\s*ID)"
        r"\s*[:#-]?\s*([A-Za-z0-9_-]+)",

        r"\b(\d{6})[_\s-]+[A-Za-z]+"
    ]

    for pattern in patterns:
        match = re.search(pattern, text, re.I)

        if match:
            return normalize_candidate_id(match.group(1))

    return ""


def extract_name(text):
    patterns = [
        r"(?:Candidate\s*Name|Applicant\s*Name|Full\s*Name|Name)"
        r"\s*[:\-]\s*([A-Za-z]+(?:\s+[A-Za-z]+){1,4})",

        r"\b\d{6}[_\s-]+([A-Za-z]+(?:[_\s-]+[A-Za-z]+){1,5})"
    ]

    for pattern in patterns:
        match = re.search(pattern, text, re.I)

        if match:
            return re.sub(
                r"[_-]+", " ", match.group(1)
            ).strip()

    return ""


def extract_email(text):
    matches = re.findall(
        r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}",
        text
    )

    return matches[0].lower() if matches else ""


def extract_phone(text):
    matches = re.findall(
        r"(?:\+91[\s-]?)?[6-9]\d{9}",
        text
    )

    return normalize_phone(matches[0]) if matches else ""


# ============================================================
# JOB ROLE FROM FOLDER
# ============================================================

def clean_role_name(folder_name):
    role = folder_name.strip()

    # Handle folders such as:
    # PE(AI-CS)-JIT14-Uniq Id
    # PE(CS-Audit & Oper)-JIT14-Uniq Id
    bracket_match = re.match(
        r"^(?:PE|PA)\s*\(([^)]+)\)",
        role,
        re.IGNORECASE
    )

    if bracket_match:
        role = bracket_match.group(1)

    else:
        # Remove PE- or PA- prefix
        role = re.sub(
            r"^(?:PE|PA)[\s_-]+",
            "",
            role,
            flags=re.IGNORECASE
        )

    # Remove Uniq Id suffix
    role = re.sub(
        r"[\s_-]*Uniq[\s_-]*Id\s*$",
        "",
        role,
        flags=re.IGNORECASE
    )

    # Remove JIT14, J14, JIT15, etc.
    role = re.sub(
        r"[\s_-]*(?:JIT|J)\s*\d+\s*$",
        "",
        role,
        flags=re.IGNORECASE
    )

    # Remove Scrn suffix
    role = re.sub(
        r"[\s_-]*Scrn\s*$",
        "",
        role,
        flags=re.IGNORECASE
    )

    # Remove Profiles or Resumes suffix
    role = re.sub(
        r"[\s_-]*(?:Profiles|Resumes)\s*$",
        "",
        role,
        flags=re.IGNORECASE
    )

    return re.sub(
        r"\s+",
        " ",
        role
    ).strip(" -_\t")


def get_job_role(file_path, root_folder):
    try:
        relative = Path(file_path).relative_to(
            Path(root_folder)
        )

        parent_folders = relative.parts[:-1]

        for folder in parent_folders:

            folder_lower = folder.lower()

            if folder_lower in GENERIC_FOLDERS:
                continue

            if re.fullmatch(
                r"(upload|archive)_\d+",
                folder_lower
            ):
                continue

            role = clean_role_name(folder)

            if role:
                return role

    except Exception:
        pass

    return "Unassigned Role"


# ============================================================
# SPREADSHEET COLUMN IDENTIFICATION
# ============================================================

ALIASES = {
    "candidate_id": [
        "candidate id", "candidate code",
        "registration id", "student id", "roll no",
        "roll number", "id"
    ],

    "name": [
        "candidate name", "applicant name", "full name",
        "name", "student name"
    ],

    "email": [
        "email", "email id", "email address", "mail", "mail id"
    ],

    "phone": [
        "phone", "phone number", "mobile", "mobile number",
        "contact", "contact number", "telephone"
    ],

    "job_role": [
        "job role", "role", "position", "job title",
        "applied role", "applied for", "designation",
        "profile", "job", "job name", "vacancy", "post name",
        "post", "post applied"
    ],

    "application_id": [
        "applicant id", "application id", "application no",
        "application number"
    ],

    "job_id": [
        "job id", "job code", "requisition id",
        "vacancy id", "position id", "opening id", "post id"
    ]
}


def find_header_row(raw):
    """Some sheets have a title row above the real headers
    (e.g. 'Centralized Notification'). Find the real header row."""
    wanted = {
        "email", "email id", "email address", "candidate id",
        "applicant id", "full name", "name", "mobile number",
        "mobile", "phone", "phone number"
    }

    for i in range(min(len(raw), 15)):
        cells = {
            re.sub(r"[^a-z0-9]+", " ", str(v).lower()).strip()
            for v in raw.iloc[i].tolist()
        }

        if len(cells & wanted) >= 2:
            return i

    return 0


def apply_header_row(raw):
    raw = raw.copy()
    header_index = find_header_row(raw)

    columns = [
        str(c).strip() if str(c).strip() else f"Unnamed_{n}"
        for n, c in enumerate(raw.iloc[header_index].tolist())
    ]

    data = raw.iloc[header_index + 1:].reset_index(drop=True)
    data.columns = columns

    return data, header_index


def normalize_column(value):
    return re.sub(
        r"[^a-z0-9]+", " ", str(value).lower()
    ).strip()


def identify_column(column):
    normalized = normalize_column(column)

    for field, aliases in ALIASES.items():

        if normalized in {
            normalize_column(a) for a in aliases
        }:
            return field

    return None


# ============================================================
# STANDARD RECORD
# ============================================================

def make_record(
    candidate_id="",
    name="",
    email="",
    phone="",
    job_role="Unassigned Role",
    job_id="",
    application_id="",
    file_name="",
    file_path="",
    file_hash="",
    resume_text="",
    source_type="Resume"
):
    return {
        "candidate_id": normalize_candidate_id(candidate_id),
        "name": clean_cell(name),
        "email": normalize_email(email),
        "phone": normalize_phone(phone),
        "job_role": clean_cell(job_role) or "Unassigned Role",
        "job_id": clean_cell(job_id),
        "application_id": clean_cell(application_id),
        "file_name": clean_cell(file_name),
        "file_path": str(file_path),
        "file_hash": file_hash,
        "resume_text": resume_text or "",
        "text_extracted": bool((resume_text or "").strip()),
        "source_type": source_type
    }


# ============================================================
# PROCESS EXCEL AND CSV
# ============================================================

def process_table_file(file_path, root_folder):
    records, errors = [], []

    name = Path(file_path).name
    extension = Path(file_path).suffix.lower()

    try:
        if extension in {".xlsx", ".xls"}:

            engine = (
                "xlrd" if extension == ".xls"
                else "openpyxl"
            )

            raw_sheets = pd.read_excel(
                file_path,
                sheet_name=None,
                header=None,
                dtype=str,
                keep_default_na=False,
                engine=engine
            )

            sheets = {}
            header_offsets = {}

            for sheet_title, raw in raw_sheets.items():
                if raw.empty:
                    continue

                data, header_index = apply_header_row(raw)
                sheets[sheet_title] = data
                header_offsets[sheet_title] = header_index

        elif extension == ".csv":

            dataframe = pd.read_csv(
                file_path,
                dtype=str,
                keep_default_na=False,
                encoding="utf-8-sig"
            )

            sheets = {"CSV": dataframe}
            header_offsets = {"CSV": 0}

        else:
            return [], []

        folder_role = get_job_role(
            file_path,
            root_folder
        )

        for sheet_name, dataframe in sheets.items():

            if dataframe.empty:
                continue

            mapping = {}

            for column in dataframe.columns:
                field = identify_column(column)

                if field and field not in mapping:
                    mapping[field] = column

            if not any(
                field in mapping
                for field in [
                    "candidate_id", "name", "email", "phone"
                ]
            ):
                errors.append({
                    "File": name,
                    "Error": (
                        f"No candidate columns found in {sheet_name}"
                    )
                })

                continue

            first_row = header_offsets.get(sheet_name, 0) + 2

            for row_number, (_, row) in enumerate(
                dataframe.iterrows(),
                start=first_row
            ):

                def value(field):
                    column = mapping.get(field)

                    return (
                        clean_cell(row.get(column, ""))
                        if column else ""
                    )

                candidate_id = value("candidate_id")
                candidate_name = value("name")
                email = value("email")
                phone = value("phone")
                spreadsheet_role = value("job_role")

                role = (
                    folder_role
                    if folder_role != "Unassigned Role"
                    else spreadsheet_role or "Unassigned Role"
                )

                job_id = value("job_id")
                application_id = value("application_id")

                if not any([
                    candidate_id,
                    candidate_name,
                    email,
                    phone
                ]):
                    continue

                row_label = (
                    f"{name} | {sheet_name} | Row {row_number}"
                )

                records.append(make_record(
                    candidate_id=candidate_id,
                    name=candidate_name,
                    email=email,
                    phone=phone,
                    job_role=role,
                    job_id=job_id,
                    application_id=application_id,
                    file_name=row_label,
                    file_path=row_label,
                    source_type="Excel/CSV"
                ))

    except Exception as error:
        errors.append({
            "File": name,
            "Error": str(error)
        })

    return records, errors


# ============================================================
# PROCESS RESUME FILE
# ============================================================

def process_resume_file(file_path, root_folder, enable_ocr):
    errors = []

    try:
        text, error = extract_text(
            file_path,
            enable_ocr=enable_ocr
        )

        if error:
            errors.append({
                "File": Path(file_path).name,
                "Error": error
            })

        source = text + "\n" + Path(file_path).stem

        record = make_record(
            candidate_id=extract_candidate_id(source),
            name=extract_name(text),
            email=extract_email(text),
            phone=extract_phone(text),
            job_role=get_job_role(file_path, root_folder),
            file_name=Path(file_path).name,
            file_path=file_path,
            file_hash=get_file_hash(file_path),
            resume_text=text,
            source_type="Resume"
        )

        return record, errors

    except Exception as error:
        errors.append({
            "File": Path(file_path).name,
            "Error": str(error)
        })

        return None, errors


# ============================================================
# SAFE ZIP EXTRACTION
# ============================================================

def extract_zip_safely(zip_file, destination):
    extracted = []

    destination = Path(destination).resolve()

    with zipfile.ZipFile(zip_file, "r") as archive:

        for item in archive.infolist():

            if item.is_dir():
                continue

            member = Path(item.filename)

            if member.is_absolute() or ".." in member.parts:
                continue

            target = (destination / member).resolve()

            if not target.is_relative_to(destination):
                continue

            if target.suffix.lower() not in SUPPORTED_FILES:
                continue

            target.parent.mkdir(
                parents=True,
                exist_ok=True
            )

            with archive.open(item) as source:
                with open(target, "wb") as output:
                    shutil.copyfileobj(source, output)

            extracted.append(target)

    return extracted


# ============================================================
# PROCESS FOLDER AND UPLOADS
# ============================================================

def process_folder(root_folder, enable_ocr):
    records, errors = [], []

    root_folder = Path(root_folder)

    if not root_folder.exists():
        return [], [{
            "File": str(root_folder),
            "Error": (
                "The configured resumes folder does not exist."
            )
        }]

    files = sorted(
        f for f in root_folder.rglob("*")
        if f.is_file()
        and f.suffix.lower() in SUPPORTED_FILES
    )

    for file_path in files:

        extension = file_path.suffix.lower()

        if extension in {".xlsx", ".xls", ".csv"}:

            found, file_errors = process_table_file(
                file_path,
                root_folder
            )

            records.extend(found)
            errors.extend(file_errors)

        elif extension in {
            ".pdf", ".doc", ".docx", ".txt"
        }:

            record, file_errors = process_resume_file(
                file_path,
                root_folder,
                enable_ocr
            )

            if record:
                records.append(record)

            errors.extend(file_errors)

    return records, errors


def process_uploads(uploaded_files, enable_ocr):
    if not uploaded_files:
        return [], []

    with tempfile.TemporaryDirectory() as temp_dir:

        root = Path(temp_dir) / "uploaded_files"
        root.mkdir(parents=True, exist_ok=True)

        all_errors = []

        for index, uploaded_file in enumerate(uploaded_files):

            name = Path(uploaded_file.name).name
            extension = Path(name).suffix.lower()

            if extension == ".zip":

                zip_path = root / f"archive_{index}.zip"

                zip_path.write_bytes(
                    uploaded_file.getvalue()
                )

                extract_dir = root / f"archive_{index}"
                extract_dir.mkdir(exist_ok=True)

                try:
                    extract_zip_safely(
                        zip_path,
                        extract_dir
                    )

                except Exception as error:
                    all_errors.append({
                        "File": name,
                        "Error": (
                            f"ZIP extraction failed: {error}"
                        )
                    })

            elif extension in SUPPORTED_FILES:

                target = root / f"upload_{index}" / name

                target.parent.mkdir(
                    parents=True,
                    exist_ok=True
                )

                target.write_bytes(
                    uploaded_file.getvalue()
                )

            else:
                all_errors.append({
                    "File": name,
                    "Error": "Unsupported file format."
                })

        records, errors = process_folder(
            root,
            enable_ocr
        )

        return records, all_errors + errors


# ============================================================
# GROUP RECORDS BELONGING TO THE SAME CANDIDATE
# ============================================================

class UnionFind:

    def __init__(self, size):
        self.parent = list(range(size))
        self.rank = [0] * size

    def find(self, x):

        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])

        return self.parent[x]

    def union(self, x, y):
        x, y = self.find(x), self.find(y)

        if x == y:
            return

        if self.rank[x] < self.rank[y]:
            self.parent[x] = y

        elif self.rank[x] > self.rank[y]:
            self.parent[y] = x

        else:
            self.parent[y] = x
            self.rank[x] += 1


def build_candidate_groups(records):
    if not records:
        return []

    uf = UnionFind(len(records))
    identity_map = {}

    for index, record in enumerate(records):

        keys = []

        if record["candidate_id"]:
            keys.append(("id", record["candidate_id"]))

        if record["email"]:
            keys.append(("email", record["email"]))

        if record["phone"]:
            keys.append(("phone", record["phone"]))

        for key in keys:

            if key in identity_map:
                uf.union(
                    index,
                    identity_map[key]
                )

            else:
                identity_map[key] = index

    groups = defaultdict(list)

    for index, record in enumerate(records):
        groups[uf.find(index)].append(record)

    return list(groups.values())


# ============================================================
# JOB APPLICATION IDENTIFICATION
# ============================================================

def get_job_key(record):
    # One notice/advertisement can share the same Job ID for many
    # posts, so the post (role) name must be part of the key.
    job_id = normalize_text(record["job_id"])
    role = normalize_text(record["job_role"])

    if role and role != "unassigned role":
        return ("job", job_id, role)

    if job_id:
        return ("job", job_id, "")

    return None


def get_job_label(record):
    return record["job_role"]


# ============================================================
# CANDIDATE JOB SUMMARY
# ============================================================

def candidate_job_application_summary(groups):
    rows = []

    for group in groups:

        ids = sorted({
            r["candidate_id"]
            for r in group
            if r["candidate_id"]
        })

        names = sorted({
            r["name"]
            for r in group
            if r["name"]
        })

        emails = sorted({
            r["email"]
            for r in group
            if r["email"]
        })

        phones = sorted({
            r["phone"]
            for r in group
            if r["phone"]
        })

        jobs = {}

        for record in group:
            key = get_job_key(record)

            if key:
                jobs[key] = get_job_label(record)

        job_labels = sorted(
            set(jobs.values()),
            key=str.lower
        )

        roles = sorted({
            r["job_role"]
            for r in group
            if r["job_role"]
            and r["job_role"].lower() != "unassigned role"
        }, key=str.lower)

        rows.append({
            "Candidate ID": ", ".join(ids) or "Not Found",
            "Candidate Name": ", ".join(names) or "Not Found",
            "Email": ", ".join(emails) or "Not Found",
            "Phone": ", ".join(phones) or "Not Found",
            "Applied Jobs": (
                ", ".join(job_labels) or "Not Available"
            ),
            "Applied Roles": (
                ", ".join(roles) or "Role Not Available"
            ),
            "Number of Jobs Applied": len(jobs),
            "Application Records": sum(
                r["source_type"] == "Excel/CSV"
                for r in group
            ),
            "Total Records": len(group),
            "Resume Records": sum(
                r["source_type"] == "Resume"
                for r in group
            )
        })

    result = pd.DataFrame(rows)

    if not result.empty:
        result = result.sort_values(
            ["Number of Jobs Applied", "Candidate Name"],
            ascending=[False, True]
        ).reset_index(drop=True)

    return result


def make_candidate_master(groups):
    summary = candidate_job_application_summary(groups)

    return summary.drop(
        columns=[
            "Application Records",
            "Resume Records"
        ],
        errors="ignore"
    )


# ============================================================
# CANDIDATE DUPLICATE SUMMARY
# ============================================================

def candidate_duplicate_summary(groups):
    rows = []

    for group in groups:

        ids = sorted({
            r["candidate_id"]
            for r in group
            if r["candidate_id"]
        })

        emails = sorted({
            r["email"]
            for r in group
            if r["email"]
        })

        roles = sorted({
            r["job_role"]
            for r in group
            if r["job_role"]
            and r["job_role"].lower() != "unassigned role"
        }, key=str.lower)

        job_counts = Counter(
            get_job_key(r)
            for r in group
            if get_job_key(r) is not None
        )

        duplicate_count = sum(
            count - 1
            for count in job_counts.values()
        )

        rows.append({
            "Candidate ID": ", ".join(ids) or "Not Found",
            "Email": ", ".join(emails) or "Not Found",
            "Duplicate Count": duplicate_count,
            "Total Roles Applied": len(roles),
            "Roles Applied": (
                ", ".join(roles) or "Unassigned Role"
            )
        })

    result = pd.DataFrame(rows)

    if not result.empty:
        result = result.sort_values(
            ["Duplicate Count", "Total Roles Applied"],
            ascending=[False, False]
        ).reset_index(drop=True)

    return result


# ============================================================
# APPLICATIONS DATAFRAME
# ============================================================

def records_dataframe(records):
    rows = []

    counts = {
        field: Counter(
            r[field].lower()
            for r in records
            if r.get(field)
        )
        for field in [
            "candidate_id", "email", "phone", "name"
        ]
    }

    for record in records:

        reasons = []

        for field, label in [
            ("candidate_id", "Duplicate ID"),
            ("email", "Duplicate Email"),
            ("phone", "Duplicate Phone")
        ]:

            value = (
                record.get(field) or ""
            ).lower()

            if value and counts[field][value] > 1:
                reasons.append(label)

        name = (
            record.get("name") or ""
        ).lower()

        if name and counts["name"][name] > 1:
            reasons.append("Repeated Name - Review")

        rows.append({
            "Candidate ID": record["candidate_id"],
            "Candidate Name": record["name"],
            "Email": record["email"],
            "Phone": record["phone"],
            "Job Role": record["job_role"],
            "Job ID": record["job_id"],
            "Source": record["source_type"],
            "File": record["file_name"],
            "Duplicate Status": (
                ", ".join(reasons) or "Unique"
            ),
            "Text Extracted": (
                "Yes" if record["text_extracted"] else "No"
            ),
            "File Path": record["file_path"]
        })

    return pd.DataFrame(rows)


# ============================================================
# DUPLICATE REPORTS
# ============================================================

DUPLICATE_VIEW_COLUMNS = [
    "Candidate Name",
    "Candidate ID",
    "Number of Jobs Applied",
    "Positions Applied"
]


def duplicate_view_row(group):
    """One simple row: name, ID, number of positions, positions."""
    summary = candidate_job_application_summary([group])

    if summary.empty:
        return None

    row = summary.iloc[0]

    return {
        "Candidate Name": row["Candidate Name"],
        "Candidate ID": row["Candidate ID"],
        "Number of Jobs Applied": row["Number of Jobs Applied"],
        "Positions Applied": row["Applied Jobs"]
    }


def sort_duplicate_view(rows):
    result = pd.DataFrame(rows, columns=DUPLICATE_VIEW_COLUMNS)

    if not result.empty:
        result = result.sort_values(
            ["Number of Jobs Applied", "Candidate Name"],
            ascending=[False, True]
        ).reset_index(drop=True)

    return result


def duplicate_value_report(records, field, title):
    grouped = defaultdict(list)

    for record in records:

        value = (
            record.get(field) or ""
        ).strip().lower()

        if value:
            grouped[value].append(record)

    rows = []

    for value, group in grouped.items():

        if len(group) <= 1:
            continue

        row = duplicate_view_row(group)

        if row:
            rows.append(row)

    return sort_duplicate_view(rows)


def exact_duplicate_report(records):
    grouped = defaultdict(list)

    for record in records:

        if record["file_hash"]:
            grouped[record["file_hash"]].append(record)

    rows = []

    for file_hash, group in grouped.items():

        if len(group) > 1:
            rows.append({
                "File Hash": file_hash,
                "Job Roles": ", ".join({
                    r["job_role"]
                    for r in group
                }),
                "Files": ", ".join(
                    r["file_name"]
                    for r in group
                ),
                "Count": len(group)
            })

    return pd.DataFrame(rows)


def multiple_record_report(groups):
    rows = []

    for group in groups:

        if len(group) <= 1:
            continue

        row = duplicate_view_row(group)

        if row:
            rows.append(row)

    return sort_duplicate_view(rows)


def repeat_application_report(groups):
    """Every application after a candidate's first one."""
    rows = []

    for group in groups:

        if len(group) <= 1:
            continue

        for number, record in enumerate(group[1:], start=2):
            rows.append({
                "Candidate Name": record["name"] or "Not Found",
                "Candidate ID": record["candidate_id"] or "Not Found",
                "Position Applied": record["job_role"],
                "Application No.": f"{number} of {len(group)}",
                "Total Applications": len(group),
                "Applicant ID": record["application_id"],
                "Source": record["file_name"],
                "_n": number
            })

    result = pd.DataFrame(rows)

    if not result.empty:
        result = result.sort_values(
            ["Total Applications", "Candidate Name", "_n"],
            ascending=[False, True, True]
        ).drop(columns="_n").reset_index(drop=True)

    return result


# ============================================================
# SIMILAR RESUME REPORT
# ============================================================

def similar_resume_report(records):
    valid = [
        r for r in records
        if r["source_type"] == "Resume"
        and r["resume_text"].strip()
    ]

    if len(valid) < 2:
        return pd.DataFrame()

    try:
        vectorizer = TfidfVectorizer(
            stop_words="english",
            max_features=10000
        )

        matrix = vectorizer.fit_transform([
            r["resume_text"]
            for r in valid
        ])

        similarity = cosine_similarity(matrix)

    except Exception:
        return pd.DataFrame()

    rows = []

    for i in range(len(valid)):
        for j in range(i + 1, len(valid)):

            score = similarity[i][j]

            if score >= SIMILARITY_THRESHOLD:
                rows.append({
                    "Candidate 1": valid[i]["name"],
                    "Candidate 2": valid[j]["name"],
                    "Resume 1": valid[i]["file_name"],
                    "Resume 2": valid[j]["file_name"],
                    "Role 1": valid[i]["job_role"],
                    "Role 2": valid[j]["job_role"],
                    "Similarity Percentage": round(
                        score * 100, 2
                    )
                })

    return pd.DataFrame(rows)


# ============================================================
# ROLE-WISE APPLICATION REPORT
# ============================================================

def role_applications_report(applications):
    if applications.empty:
        return pd.DataFrame()

    data = applications[
        applications["Job Role"] != "Unassigned Role"
    ]

    return (
        data.groupby("Job Role")
        .agg(
            Total_Application_Records=(
                "Candidate Name", "count"
            ),
            Unique_Candidate_Emails=(
                "Email", "nunique"
            )
        )
        .reset_index()
        .rename(columns={
            "Total_Application_Records": (
                "Total Application Records"
            ),
            "Unique_Candidate_Emails": (
                "Unique Candidate Emails"
            )
        })
    )


# ============================================================
# DISPLAY HELPERS
# ============================================================

def show_dataframe(title, dataframe):
    st.subheader(title)

    if dataframe is None or dataframe.empty:
        st.info("No records found.")

    else:
        st.dataframe(
            dataframe,
            use_container_width=True,
            hide_index=True
        )


def highlight_duplicates(dataframe):
    if dataframe.empty:
        st.info("No records found.")
        return

    def style_row(row):
        status = str(
            row.get("Duplicate Status", "")
        )

        if "Duplicate" in status:
            return [
                "background-color: #ffcdd2; color: #8b0000"
            ] * len(row)

        if "Review" in status:
            return [
                "background-color: #fff3cd; color: #856404"
            ] * len(row)

        return [""] * len(row)

    st.dataframe(
        dataframe.style.apply(
            style_row,
            axis=1
        ),
        use_container_width=True,
        hide_index=True
    )


# ============================================================
# EXCEL EXPORT
# ============================================================

def to_excel_bytes(sheets):
    output = BytesIO()

    with pd.ExcelWriter(
        output,
        engine="openpyxl"
    ) as writer:

        written = False

        for sheet_name, dataframe in sheets.items():

            if dataframe is not None:

                dataframe.to_excel(
                    writer,
                    sheet_name=sheet_name[:31],
                    index=False
                )

                written = True

        if not written:
            pd.DataFrame({
                "Message": [
                    "No records found."
                ]
            }).to_excel(
                writer,
                sheet_name="No Records",
                index=False
            )

    output.seek(0)

    return output.getvalue()


# ============================================================
# MAIN APPLICATION
# ============================================================

def main():
    create_folders()

    st.title(
        "📄 Resume Duplicate Tracking System"
    )

    st.markdown("""
    Upload candidate resumes, spreadsheets and ZIP folders.

    Automatically extract candidate details, identify duplicates,
    and track how many jobs each candidate has applied for.
    """)

    st.sidebar.header("Analysis Settings")

    enable_ocr = st.sidebar.checkbox(
        "Enable OCR for scanned PDFs",
        value=True,
        help="Requires Tesseract OCR to be installed."
    )

    scan_folder = st.sidebar.checkbox(
        "Also scan the configured resumes folder",
        value=False,
        help=(
            "OFF = analyze only the files you upload (recommended). "
            "ON = also include every file inside the resumes folder."
        )
    )

    st.sidebar.write(
        f"Resume similarity threshold: "
        f"{SIMILARITY_THRESHOLD * 100:.0f}%"
    )

    st.sidebar.caption(
        "Files inside the configured resumes folder "
        "are scanned automatically."
    )

    st.sidebar.caption(
        f"Folder: {RESUME_FOLDER}"
    )

    st.divider()

    st.subheader("Upload Candidate Files")

    uploaded_files = st.file_uploader(
        "Upload multiple files or ZIP folders",
        type=[
            "pdf", "doc", "docx", "xls", "xlsx",
            "csv", "txt", "zip"
        ],
        accept_multiple_files=True
    )

    st.caption(
        "Supported: PDF, DOC, DOCX, XLS, XLSX, CSV, TXT and ZIP."
    )

    st.info(
        "You can also place files inside the resumes folder. "
        "The application scans its subfolders automatically."
    )

    st.write(
        f"Files selected: {len(uploaded_files or [])}"
    )

    st.caption(
        "Note: a ZIP counts as 1 selected file but can contain "
        "many records."
    )

    if st.button(
        "🔍 Analyze Files and Detect Duplicates",
        type="primary",
        use_container_width=True
    ):

        with st.spinner(
            "Extracting and analyzing candidate records..."
        ):

            folder_records, folder_errors = [], []

            if scan_folder:
                folder_records, folder_errors = process_folder(
                    RESUME_FOLDER,
                    enable_ocr
                )

            upload_records, upload_errors = process_uploads(
                uploaded_files or [],
                enable_ocr
            )

            all_records = folder_records + upload_records

            # Skip anything counted twice:
            #  - same resume file (same content, same role)
            #  - same Excel application (same Applicant ID, same role)
            seen, unique_records = set(), []

            for rec in all_records:
                key = None

                if rec["file_hash"]:
                    key = ("file", rec["file_hash"], rec["job_role"])

                elif rec["application_id"]:
                    key = (
                        "app",
                        rec["application_id"].lower(),
                        rec["job_role"].lower()
                    )

                if key is not None:
                    if key in seen:
                        continue

                    seen.add(key)

                unique_records.append(rec)

            st.session_state["removed_duplicates"] = (
                len(all_records) - len(unique_records)
            )

            st.session_state["analysis_records"] = unique_records

            st.session_state["analysis_errors"] = (
                folder_errors + upload_errors
            )

            st.session_state["analysis_done"] = True

    if not st.session_state.get(
        "analysis_done",
        False
    ):

        st.info(
            "Upload files or add them to the resumes folder, "
            "then click Analyze Files."
        )

        return

    records = st.session_state["analysis_records"]
    errors = st.session_state["analysis_errors"]

    if not records:

        st.warning(
            "No candidate records were extracted. "
            "Check your uploaded files and spreadsheet columns."
        )

        if errors:
            show_dataframe(
                "Processing Errors",
                pd.DataFrame(errors)
            )

        return

    # ========================================================
    # ANALYSIS
    # ========================================================

    groups = build_candidate_groups(records)

    master = make_candidate_master(groups)

    job_summary = candidate_job_application_summary(
        groups
    )

    candidate_duplicates = candidate_duplicate_summary(
        groups
    )

    applications = records_dataframe(records)

    role_report = role_applications_report(
        applications
    )

    multiple_records = multiple_record_report(
        groups
    )

    duplicate_ids = duplicate_value_report(
        records,
        "candidate_id",
        "Candidate ID"
    )

    duplicate_emails = duplicate_value_report(
        records,
        "email",
        "Email"
    )

    duplicate_phones = duplicate_value_report(
        records,
        "phone",
        "Phone"
    )

    duplicate_names = duplicate_value_report(
        records,
        "name",
        "Candidate Name"
    )

    exact_duplicates = exact_duplicate_report(
        records
    )

    repeat_applications = repeat_application_report(
        groups
    )

    similar_resumes = similar_resume_report(
        records
    )

    # ========================================================
    # DASHBOARD
    # ========================================================

    st.divider()

    st.header("📊 Dashboard Overview")

    resume_count = sum(
        r["source_type"] == "Resume"
        for r in records
    )

    table_count = sum(
        r["source_type"] == "Excel/CSV"
        for r in records
    )

    total_roles = applications["Job Role"].nunique()

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Total Records", len(records))
    c2.metric("Resume Records", resume_count)
    c3.metric("Excel/CSV Records", table_count)
    c4.metric("Unique Candidates", len(groups))

    c5, c6, c7, c8 = st.columns(4)

    c5.metric("Job Roles", total_roles)
    c6.metric("Repeated IDs", len(duplicate_ids))
    c7.metric("Repeated Emails", len(duplicate_emails))
    c8.metric("Repeated Phones", len(duplicate_phones))

    c9, c10, c11, c12 = st.columns(4)

    c9.metric(
        "Repeat Applications",
        len(repeat_applications)
    )
    c10.metric(
        "Candidates With Repeats",
        int((job_summary["Total Records"] > 1).sum())
    )
    c11.metric(
        "Double-Loaded Rows Removed",
        st.session_state.get("removed_duplicates", 0)
    )

    if errors:
        with st.expander(
            f"⚠️ Processing Errors ({len(errors)})"
        ):

            show_dataframe(
                "Files That Need Attention",
                pd.DataFrame(errors)
            )

    # ========================================================
    # MAIN TABS
    # ========================================================

    st.divider()

    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "Candidate Master List",
        "All Applications",
        "Role-wise Applications",
        "Multi-Job Candidates",
        "Candidate Job Summary",
        "Candidate Duplicate Summary"
    ])

    with tab1:
        show_dataframe(
            "Unique Candidate Master List",
            master
        )

    with tab2:
        st.markdown(
            "Red rows indicate repeated IDs, emails or phone numbers. "
            "Repeated names require manual review."
        )

        highlight_duplicates(applications)

    with tab3:
        show_dataframe(
            "Role-wise Application Report",
            role_report
        )

    with tab4:
        multi_role = job_summary[
            job_summary["Number of Jobs Applied"] > 1
        ].copy()

        show_dataframe(
            "Candidates Applying for Multiple Jobs",
            multi_role
        )

    with tab5:
        show_dataframe(
            "Candidate-wise Job Application Summary",
            job_summary
        )

        st.caption(
            "Distinct Job IDs are counted separately. When no Job ID "
            "exists, distinct job role names are used. Multiple files "
            "for the same role are counted once per candidate."
        )

    with tab6:
        show_dataframe(
            "Candidate Duplicate Summary",
            candidate_duplicates
        )

        st.caption(
            "Duplicate Count shows extra records for an already "
            "represented job. Applications to different roles "
            "are not duplicates."
        )

    # ========================================================
    # DUPLICATE DETECTION
    # ========================================================

    st.divider()

    st.header("🔎 Duplicate Detection Reports")

    d0, d1, d2, d3, d4, d5, d6 = st.tabs([
        "Repeat Applications",
        "Multiple Records",
        "Repeated IDs",
        "Repeated Emails",
        "Repeated Phones",
        "Repeated Names",
        "Similar Resumes"
    ])

    with d0:
        show_dataframe(
            "Repeat Applications (every application after a "
            "candidate's first)",
            repeat_applications
        )

        st.caption(
            "Each row is an extra application by a candidate who "
            "already appears once. Total Applications is how many "
            "records that candidate has."
        )

    with d1:
        show_dataframe(
            "Multiple Linked Candidate Records",
            multiple_records
        )

    with d2:
        show_dataframe(
            "Repeated Candidate IDs",
            duplicate_ids
        )

    with d3:
        show_dataframe(
            "Repeated Email Addresses",
            duplicate_emails
        )

    with d4:
        show_dataframe(
            "Repeated Phone Numbers",
            duplicate_phones
        )

    with d5:
        show_dataframe(
            "Repeated Candidate Names",
            duplicate_names
        )

    with d6:
        show_dataframe(
            "Similar Resume Content",
            similar_resumes
        )

    # ========================================================
    # DOWNLOAD DUPLICATE REPORTS
    # ========================================================

    st.divider()

    st.header("📥 Download Duplicate Records")

    all_candidates = job_summary[[
        "Candidate Name", "Candidate ID",
        "Number of Jobs Applied", "Applied Jobs"
    ]].rename(columns={"Applied Jobs": "Positions Applied"})

    duplicate_sheets = {
        "All Candidates": all_candidates,
        "Repeat Applications": repeat_applications,
        "Multiple Records": multiple_records,
        "Repeated IDs": duplicate_ids,
        "Repeated Emails": duplicate_emails,
        "Repeated Phones": duplicate_phones,
        "Repeated Names": duplicate_names,
        "Exact Duplicates": exact_duplicates,
        "Similar Resumes": similar_resumes
    }

    duplicate_excel_data = to_excel_bytes(
        duplicate_sheets
    )

    st.download_button(
        "📊 Download All Duplicate Records (Excel)",
        data=duplicate_excel_data,
        file_name="Complete_Duplicate_Records.xlsx",
        mime=(
            "application/vnd.openxmlformats-officedocument."
            "spreadsheetml.sheet"
        ),
        type="primary",
        use_container_width=True
    )

    # ========================================================
    # CANDIDATE SEARCH
    # ========================================================

    st.divider()

    st.header("🔍 Search Candidate")

    search_value = st.text_input(
        "Enter candidate name, email, phone or ID"
    )

    if search_value.strip():

        search = search_value.strip().lower()

        searchable = job_summary.astype(str)

        mask = searchable.apply(
            lambda column: column.str.lower().str.contains(
                re.escape(search),
                na=False
            )
        ).any(axis=1)

        show_dataframe(
            "Candidate Search Results",
            job_summary[mask]
        )

    # ========================================================
    # DOWNLOAD COMPLETE REPORTS
    # ========================================================

    st.divider()

    st.header("📥 Download Complete Reports")

    sheets = {
        "Candidate Master": master,
        "All Applications": applications,
        "Role Applications": role_report,
        "Candidate Job Summary": job_summary,
        "Candidate Duplicate Summary": candidate_duplicates,
        "Repeat Applications": repeat_applications,
        "Multiple Records": multiple_records,
        "Repeated IDs": duplicate_ids,
        "Repeated Emails": duplicate_emails,
        "Repeated Phones": duplicate_phones,
        "Repeated Names": duplicate_names,
        "Exact Duplicates": exact_duplicates,
        "Similar Resumes": similar_resumes,
        "Processing Errors": pd.DataFrame(errors)
    }

    excel_data = to_excel_bytes(sheets)

    st.download_button(
        "📊 Download Complete Excel Report",
        data=excel_data,
        file_name="Resume_Duplicate_Tracking_Report.xlsx",
        mime=(
            "application/vnd.openxmlformats-officedocument."
            "spreadsheetml.sheet"
        ),
        type="primary",
        use_container_width=True
    )

    st.divider()

    st.caption(
        "Candidate matching uses candidate ID, email or phone. "
        "Repeated names alone are not treated as confirmed identity "
        "matches. Similar resume scores are indicators for manual review."
    )


if __name__ == "__main__":
    main()