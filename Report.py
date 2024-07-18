import pymupdf
import re

class Report:
    def __init__(self, path):
        self.path = path
        self.title = []
        self.date = []
        self.authors = []
        self.reportNumber = []
        self.client = []
        self.category = []
        self.jobNumber = []
        self.body = []
        self.execSum = []
        self.figures = []
        self.tables = []
        self.contents = dict()
        self.docType = self.GetDocType()

    def __repr__(self):
        # maxTitle = ""
        # for title in self.title:
        #     if len(title)>len(maxTitle):
        #         maxTitle = title
        return f"Report at: {self.path}"

    def GetDocType(self):
        if ".pdf" in self.path:
            docType = "pdf"
        elif ".doc" in self.path:
            docType = "doc"
        else:
            docType = None
        return docType

    def GetText(self):

        if self.docType == "doc":
            self.docType = None
            # pdfPath = ".".join(self.path.split(".")[:-1])+".pdf"
            # if os.path.exists(pdfPath):
            #     self.path = pdfPath
            #     self.docType = "pdf"
            # else:
            #     try:
            #         newPath = "-".join(pdfPath.split(".")[0:-1])+str("_Autoconverted.pdf")
            #         docx2pdf.convert(self.path, newPath)
            #         self.path = newPath
            #         self.docType = "pdf"
            #     except AssertionError:
            #         print("Cannot parse non .docm word files, please convert to a non macro enabled document")
            #         self.docType = None
            #         return
            #     except Exception as e:
            #         print(f"Following Exception Occured: \n {e} \n Have you made sure that the document is not open?")



        if self.docType == "pdf":
            doc = pymupdf.open(self.path)
            fullText = ""
            pages = []
            for page in doc:
                pageText = page.get_text()
                fullText += "\n NEW PAGE \n" + str(pageText)
                lines = pageText.split("\n")
                lines = [line.strip() for line in lines]
                lines = [line for line in lines if not line == ""]
                pages.append(lines)
            self.body = fullText
            print(self.path)
            return pages
        else:
            print(f"Cannot Read Document {self.path}")
            return None

    def ExtractText(self):

        buzzWords = ["prepared for", "prepared for:", "report no.", "executive summary", "table of contents"]
        continueContents = False
        doc = self.GetText()
        if doc is None:
            return
        fullText = ""
        for pageNum, page in enumerate(doc):
            pageText = page
            for line in page:
                fullText += line.strip() + "\n"
            if pageNum == 0:
                self.title.append(fullText.strip())
            if pageNum < 5:
                for word in buzzWords:
                    if word in [line.strip().lower() for line in pageText]:
                        print(word, pageNum)
                        for lineNum, line in enumerate([line.strip().lower() for line in pageText]):
                            line = line.strip()
                            if ("prepared for" in line) or ("prepared for:" in line):
                                counter = 1
                                while len(pageText[lineNum + counter].strip())>2:

                                    self.client.append(pageText[lineNum+counter])
                                    counter += 1
                                    if len(pageText) <= (lineNum + counter):
                                        break

                            if "report No." in line:
                                self.reportNumber.append(pageText[lineNum+1])
                            if "date" in line and (lineNum + 1 < len(pageText)):
                                self.date.append(pageText[lineNum+1])
                            if line == "by":
                                self.authors.extend(re.split('and |,', pageText[lineNum+1]))
                            if "executive summary" in line:
                                if "table of contents" not in [_line.lower().strip() for _line in doc[pageNum+1]]:
                                    self.execSum.append("\n".join(pageText) + "\n".join(doc[pageNum+1]))
                                else:
                                    self.execSum.append("\n".join(pageText))
            contentsStart = False
            for lineNum, line in enumerate([_line.lower().strip() for _line in pageText]):
                if "table of contents" in line:
                    continueContents = True
                    contentsStart = True
                if contentsStart:
                    for char in line:
                        if char.isalpha():
                            self.contents["Page " + str(pageNum) + "Line " + str(lineNum)] = line
                            break

            if continueContents:
                continueContents = False
                for lineNum, line in enumerate([_line.lower().strip() for _line in pageText]):
                    if "table of contents" in line:
                        continueContents = True
                    if contentsStart:
                        for char in line:
                            if char.isalpha():
                                self.contents["Page " + str(pageNum) + " Line " + str(lineNum)] = line
                                break
        self.body = fullText