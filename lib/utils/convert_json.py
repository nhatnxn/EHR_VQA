# -*- coding: utf-8 -*-

from models.cell import Cell
from models.line import Line
from models.paragraph import Paragraph
from models.table import Table
from models.analyze_result import AnalyzeResult
from lib.utils.encoder import JSONEncoder


def reference_line_by_bbx(bounding_box_cell, paragraph_result):
    index_paragraphs = []
    index_lines = []
    for index_paragraph, paragraph in enumerate(paragraph_result):
        for index_line, line in enumerate(paragraph[2]):
            bounding_box_line = [line[0][0], line[0][1], line[0][2], line[0][3]]
            if (bounding_box_line[0] > bounding_box_cell[0])\
                and (bounding_box_line[1] > bounding_box_cell[1])\
                and (bounding_box_line[2] < bounding_box_cell[2])\
                and (bounding_box_line[3] < bounding_box_cell[3]):
                index_paragraphs.append(index_paragraph)
                index_lines.append(index_line)
    
    return index_paragraphs, index_lines


def reference_line_by_text(text_cell, paragraph_result):
    index_paragraphs = []
    index_lines = []
    for index_paragraph, paragraph in enumerate(paragraph_result):
        for index_line, line in enumerate(paragraph[2]):
            if line[1] == text_cell:
                index_paragraphs.append(index_paragraph)
                index_lines.append(index_line)
    
    return index_paragraphs, index_lines


def convert_to_json(ehr_res, width, height):
    ocr_result = ehr_res["ocr_result"]
    table_result = ehr_res["table_reconstruct_result"]
    paragraph_result = ehr_res["paragraph_result"]

    paragraphs = []
    for item in paragraph_result:
        lines = []
        for line in item[2]:
            lines.append(
                JSONEncoder().default(
                    Line(bounding_box=[line[0][0], line[0][1], line[0][2], line[0][3]],
                        text=line[1], confidence=line[2])
                )
            )
        
        paragraphs.append(
            JSONEncoder().default(
                Paragraph(bounding_box=[item[0][0], item[0][1], item[0][2], item[0][3]],
                        text=item[1], lines=lines)
            )
        )

    tables = []
    for table in table_result:
        try:
            number_rows = 0
            number_cols = 0
            cells = []
            if len(table["text"]) == 0:
                continue
            
            for row_index, texts in enumerate(table["text"]):
                number_rows += 1
                for column_index, text_cell in enumerate(texts):
                    if column_index + 1 >= number_cols:
                        number_cols = column_index + 1
                    
                    index_paragraphs = None
                    index_lines = None
                    elements = None

                    bounding_box_cell = table['table_reconstructed_coordinate'][row_index][column_index]
                    if bounding_box_cell == -1:
                        bounding_box_cell = None
                    else:
                        bounding_box_cell = list(map(int, bounding_box_cell.split(",")))
                        index_paragraphs, index_lines = reference_line_by_bbx(bounding_box_cell, paragraph_result)
                        # index_paragraphs, index_lines = reference_line_by_text(text_cell, paragraph_result)
                    if index_paragraphs and index_lines:
                        elements = []
                        for index, _ in enumerate(index_paragraphs):
                            elements.append(f"#/paragraphResults/{index_paragraphs[index]}/lines/{index_lines[index]}")
                    
                    cells.append(
                        JSONEncoder().default(
                            Cell(row_index=row_index, column_index=column_index,
                                row_span=None, col_span=None, 
                                text=text_cell, 
                                bounding_box=bounding_box_cell,
                                elements=elements, is_header=None)
                        )
                    )
            
            tables.append(
                JSONEncoder().default(
                    Table(rows=number_rows, columns=number_cols,
                        bounding_box=table['table_box'],
                        cells=cells)
                )
            )
        except:
            continue

    # analyze_result
    analyze_result = AnalyzeResult(width=width, height=height, 
        paragraph_results=paragraphs, table_results=tables)
    analyze_result = JSONEncoder().default(analyze_result)

    return analyze_result
