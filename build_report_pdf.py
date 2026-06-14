# -*- coding: utf-8 -*-
"""把课程报告 Markdown 渲染为样式化 HTML(供 headless Chrome 打印 PDF)。"""
import sys, pathlib, markdown

ROOT = pathlib.Path(__file__).parent
md_path = ROOT / 'COURSE_REPORT_SingerLens_SVDD.md'
html_path = ROOT / 'COURSE_REPORT_SingerLens_SVDD.html'

text = md_path.read_text(encoding='utf-8')
body = markdown.markdown(
    text,
    extensions=['tables', 'fenced_code', 'sane_lists', 'attr_list', 'md_in_html'],
    output_format='html5',
)

CSS = """
@page { size: A4; margin: 18mm 16mm 18mm 16mm; }
html { -webkit-print-color-adjust: exact; print-color-adjust: exact; }
body {
  font-family: "SimSun", "Songti SC", "Source Han Serif SC", serif;
  font-size: 11.5pt; line-height: 1.75; color: #1a1a1a;
  max-width: 820px; margin: 0 auto; padding: 0 4px;
}
h1, h2, h3, h4 {
  font-family: "Microsoft YaHei", "PingFang SC", "Source Han Sans SC", sans-serif;
  color: #111; line-height: 1.35; page-break-after: avoid;
}
h1 { font-size: 19pt; text-align: center; margin: 6pt 0 14pt; line-height: 1.4; }
h2 { font-size: 15pt; margin: 20pt 0 8pt; padding-bottom: 4pt; border-bottom: 2px solid #2b2b2b; }
h3 { font-size: 12.5pt; margin: 14pt 0 6pt; color: #222; }
p { margin: 7pt 0; text-align: justify; }
blockquote {
  color: #555; border-left: 3px solid #c9c9c9; margin: 8pt 0;
  padding: 2pt 12pt; font-size: 10.5pt; background: #fafafa;
}
strong { color: #111; }
code { font-family: "Consolas", monospace; background: #f2f2f2; padding: 0 3px; border-radius: 3px; font-size: 10pt; }
img { max-width: 100%; display: block; margin: 8pt auto 4pt; page-break-inside: avoid; }
table {
  border-collapse: collapse; width: 100%; margin: 8pt 0; font-size: 10.3pt;
  page-break-inside: avoid;
}
th, td { border: 1px solid #b8b8b8; padding: 4pt 7pt; text-align: left; }
th { background: #eef1f4; font-family: "Microsoft YaHei", sans-serif; }
tr:nth-child(even) td { background: #fafbfc; }
hr { border: none; border-top: 1px solid #ddd; margin: 14pt 0; }
ul, ol { margin: 6pt 0; padding-left: 22pt; }
li { margin: 3pt 0; }
/* 图注(图: 开头的加粗段落)与其后的图片尽量同页 */
p > strong:first-child { }
"""

html = f"""<!DOCTYPE html>
<html lang="zh-CN"><head><meta charset="utf-8">
<title>SVDD 评测混杂因子审计</title>
<style>{CSS}</style>
</head><body>
{body}
</body></html>"""

html_path.write_text(html, encoding='utf-8')
print('HTML ->', html_path)
print('chars:', len(html))
