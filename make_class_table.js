const {
  Document, Packer, Paragraph, TextRun, Table, TableRow, TableCell,
  AlignmentType, BorderStyle, WidthType, ShadingType, VerticalAlign,
  HeadingLevel
} = require("docx");
const fs = require("fs");

// Tab20 hex colors for classes 0–10 (matplotlib tab20 palette)
const TAB20_HEX = [
  "1f77b4", // 0  Низкая растительность      — синий
  "aec7e8", // 1  Непроницаемая поверхность  — голубой
  "ff7f0e", // 2  Транспортное средство       — оранжевый
  "ffbb78", // 3  Городская мебель            — светло-оранжевый
  "2ca02c", // 4  Крыша                       — зелёный
  "98df8a", // 5  Фасад                       — светло-зелёный
  "d62728", // 6  Кустарник                   — красный
  "ff9896", // 7  Дерево                      — розовый
  "9467bd", // 8  Грунт/гравий               — фиолетовый
  "c5b0d5", // 9  Вертикальная поверхность   — светло-фиолетовый
  "8c564b", // 10 Дымовая труба              — коричневый
];

const CLASS_DATA = [
  { id: 0,  name: "Низкая растительность",     count: 897950,  pct: 40.81 },
  { id: 1,  name: "Непроницаемая поверхность", count: 470303,  pct: 21.38 },
  { id: 2,  name: "Транспортное средство",     count: 13985,   pct:  0.64 },
  { id: 3,  name: "Городская мебель",          count: 30189,   pct:  1.37 },
  { id: 4,  name: "Крыша",                     count: 377328,  pct: 17.15 },
  { id: 5,  name: "Фасад",                     count: 28858,   pct:  1.31 },
  { id: 6,  name: "Кустарник",                 count: 102112,  pct:  4.64 },
  { id: 7,  name: "Дерево",                    count: 222200,  pct: 10.10 },
  { id: 8,  name: "Грунт/гравий",             count: 40221,   pct:  1.83 },
  { id: 9,  name: "Вертикальная поверхность",  count: 15310,   pct:  0.70 },
  { id: 10, name: "Дымовая труба",             count: 1757,    pct:  0.08 },
];

// Column widths (A4 content width = 11906 - 2*1134 ≈ 9638 DXA, margins 2.5cm each)
const PAGE_W   = 11906;
const MARGIN   = 1440; // ~2.54 cm
const CONTENT  = PAGE_W - 2 * MARGIN; // 9026 DXA

const COL_NUM   = 800;
const COL_COLOR = 1200;
const COL_PCT   = 1400;
const COL_NAME  = CONTENT - COL_NUM - COL_COLOR - COL_PCT; // остаток

const border = { style: BorderStyle.SINGLE, size: 4, color: "AAAAAA" };
const borders = { top: border, bottom: border, left: border, right: border };
const noBorder = { style: BorderStyle.NIL, size: 0, color: "FFFFFF" };
const noBorders = { top: noBorder, bottom: noBorder, left: noBorder, right: noBorder };

function cell(text, width, opts = {}) {
  return new TableCell({
    width:   { size: width, type: WidthType.DXA },
    borders,
    verticalAlign: VerticalAlign.CENTER,
    margins: { top: 80, bottom: 80, left: 120, right: 120 },
    shading: opts.fill ? { fill: opts.fill, type: ShadingType.CLEAR } : undefined,
    children: [new Paragraph({
      alignment: opts.align || AlignmentType.CENTER,
      children: [new TextRun({
        text:  text,
        bold:  !!opts.bold,
        size:  20,           // 10pt
        font:  "Times New Roman",
        color: opts.textColor || "000000",
      })],
    })],
  });
}

function headerCell(text, width) {
  return new TableCell({
    width:   { size: width, type: WidthType.DXA },
    borders,
    verticalAlign: VerticalAlign.CENTER,
    shading: { fill: "D9D9D9", type: ShadingType.CLEAR },
    margins: { top: 80, bottom: 80, left: 120, right: 120 },
    children: [new Paragraph({
      alignment: AlignmentType.CENTER,
      children: [new TextRun({
        text,
        bold: true,
        size: 20,
        font: "Times New Roman",
      })],
    })],
  });
}

// Decide text contrast (black or white) based on luminance
function contrastText(hex) {
  const r = parseInt(hex.slice(0,2),16);
  const g = parseInt(hex.slice(2,4),16);
  const b = parseInt(hex.slice(4,6),16);
  const lum = 0.299*r + 0.587*g + 0.114*b;
  return lum > 140 ? "000000" : "FFFFFF";
}

// Build table rows
const rows = [
  // Header row
  new TableRow({
    tableHeader: true,
    children: [
      headerCell("№",       COL_NUM),
      headerCell("Класс",   COL_NAME),
      headerCell("Цвет",    COL_COLOR),
      headerCell("Доля, %", COL_PCT),
    ],
  }),
];

for (const d of CLASS_DATA) {
  const hex = TAB20_HEX[d.id];
  const textColor = contrastText(hex);

  // Color cell: filled with the class color, text = hex (optional, can leave empty)
  const colorCell = new TableCell({
    width:   { size: COL_COLOR, type: WidthType.DXA },
    borders,
    verticalAlign: VerticalAlign.CENTER,
    shading: { fill: hex.toUpperCase(), type: ShadingType.CLEAR },
    margins: { top: 80, bottom: 80, left: 120, right: 120 },
    children: [new Paragraph({
      alignment: AlignmentType.CENTER,
      children: [new TextRun({
        text: `#${hex.toUpperCase()}`,
        size: 16,
        font: "Courier New",
        color: textColor,
      })],
    })],
  });

  rows.push(new TableRow({
    children: [
      cell(String(d.id), COL_NUM),
      cell(d.name,       COL_NAME, { align: AlignmentType.LEFT }),
      colorCell,
      cell(d.pct.toFixed(2), COL_PCT),
    ],
  }));
}

const table = new Table({
  width: { size: CONTENT, type: WidthType.DXA },
  columnWidths: [COL_NUM, COL_NAME, COL_COLOR, COL_PCT],
  rows,
});

const doc = new Document({
  styles: {
    default: {
      document: { run: { font: "Times New Roman", size: 24 } },
    },
  },
  sections: [{
    properties: {
      page: {
        size:   { width: PAGE_W, height: 16838 },
        margin: { top: MARGIN, right: MARGIN, bottom: MARGIN, left: MARGIN },
      },
    },
    children: [
      // Caption above the table (ГОСТ style)
      new Paragraph({
        spacing: { after: 120 },
        children: [new TextRun({
          text: "Таблица 1 – Семантические классы датасета Hessigheim 3D " +
                "(номер, название, цвет, доля точек в Mar16-train)",
          size: 20,
          font: "Times New Roman",
        })],
      }),
      table,
    ],
  }],
});

Packer.toBuffer(doc).then(buf => {
  fs.writeFileSync("results/class_table.docx", buf);
  console.log("Saved: results/class_table.docx");
});
