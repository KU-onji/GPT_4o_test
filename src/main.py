from utils import summarize_pdf_on_web


def main():
    summary = summarize_pdf_on_web(str(input("Enter arXiv URL of the PDF file: ")))
    print(summary)


if __name__ == "__main__":
    main()
