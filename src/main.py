from utils import summarize_pdf_on_web


def main():
    summarize_pdf_on_web(str(input("Enter arXiv URL of the PDF file: ")))


if __name__ == "__main__":
    main()
