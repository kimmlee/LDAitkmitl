mkdir converted_pdf
find . -name "*.pdf" -type f | while read line; do
    gs -dEmbedAllFonts=true -o converted_pdf/${line/.\//} -sDEVICE=pdfwrite "${line/.\//}"
done