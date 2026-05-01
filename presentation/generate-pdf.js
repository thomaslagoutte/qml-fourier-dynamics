// generate-pdf.js
// ---------------------------------------------------------------
// 1️⃣  Load the local HTML file (cv.html)
// 2️⃣  Tell the browser to use the print‑media stylesheet
// 3️⃣  Export a single‑page A4 PDF with tight margins
// ---------------------------------------------------------------

const puppeteer = require('puppeteer');
const path = require('path');

(async () => {
  // -----------------------------------------------------------
  // 1️⃣  Launch a headless Chrome instance
  // -----------------------------------------------------------
  const browser = await puppeteer.launch({headless: true});
  const page = await browser.newPage();

  // -----------------------------------------------------------
  // 2️⃣  Load the HTML file from the same folder as this script
  // -----------------------------------------------------------
  const htmlPath = path.resolve(__dirname, 'cv.html');   // <-- make sure the file name matches
  await page.goto(`file://${htmlPath}`, {waitUntil: 'networkidle0'});

  // -----------------------------------------------------------
  // 3️⃣  Apply the print stylesheet (Tailwind `print:` utilities)
  // -----------------------------------------------------------
  await page.emulateMediaType('print');

  // -----------------------------------------------------------
  // 4️⃣  PDF options – the key part that controls margins & scaling
  // -----------------------------------------------------------
  const pdfOptions = {
    path: path.resolve(__dirname, 'cv.pdf'),   // output file
    format: 'A4',                             // A4 paper size (210 mm × 297 mm)
    printBackground: true,                    // keep teal borders, background colours, etc.
    margin: {
      top:    '3mm',   // shrink top margin
      bottom: '4mm',   // shrink bottom margin
      left:   '9.75mm',  // shrink left margin
      right:  '9.75mm',  // shrink right margin
    },
  };

  // -----------------------------------------------------------
  // 5️⃣  Generate the PDF
  // -----------------------------------------------------------
  await page.pdf(pdfOptions);

  // -----------------------------------------------------------
  // 6️⃣  Clean up
  // -----------------------------------------------------------
  await browser.close();
  console.log('✅ PDF saved as cv.pdf');
})();