# import PyPDF2
# import re

# class DocumentParser:
#     def parse_pdf(self, file_path: str) -> tuple[str, str]:
#         text = ""
#         with open(file_path, "rb") as file:
#             reader = PyPDF2.PdfReader(file)
#             for page in reader.pages:
#                 text += page.extract_text()

#         # inline title‐extraction (formerly extract_title)
#         lines = text.split("\n")
#         candidates = []
#         for i, line in enumerate(lines[:10]):
#             clean_line = line.strip()
#             if not clean_line or len(clean_line) < 5:
#                 continue
#             score = 0
#             if re.match(r"^(CONTRACT|AGREEMENT|PETITION|NOTICE|ORDER|BILL|ACT|STATUTE)\b",
#                         clean_line, re.IGNORECASE):
#                 score += 5
#             if re.match(r"^[A-Z\s\-]{5,}$", clean_line):
#                 score += 2
#             if "**" in clean_line or clean_line.center(80) == clean_line:
#                 score += 1
#             candidates.append((clean_line, score))
#         title = candidates[0][0] if candidates else "Unknown Title"

#         return title, text
