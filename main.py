import sys
sys.path.append("..") # Adds higher directory to python modules path.

from Util import Util
from LDAModeling import LDAModeling

import os
# This package is for downloading pdf
import urllib.request

import json

"""
    1) download all files from a list of URLs and save to local (API server)
    require:
        local_path = "/Users/Kim/Documents/TestDownloadFiles/"
"""
# todo change these three variables
# define a local root to save files

input_local_root = '/Users/Kim/Documents/trf_dir/TestDownloadFiles/ori/'
# input_local_root = '/Users/dhanamon/LDAitkmitl/TestDownloadFiles/'

converted_local_root = '/Users/Kim/Documents/trf_dir/TestDownloadFiles/converted/'


# define an output directory to save an 'original' pyLDAvis html file
output_dir = '/Users/Kim/Documents/trf_dir/PyLDAVizOutput/'
# output_dir = '/Users/dhanamon/LDAitkmitl/PyLDAVizOutput/'
# define an output directory to save an 'original' pyLDAvis html file
pyLDAvis_output_file = '5pdf_LDAvis_newmm_2n_postag_title_10n.html'

# define an output directory to save an 'thai' pyLDAvis html file
th_output_dir = '/Users/Kim/Documents/trf_dir/PyLDAVizOutput/th/'
# th_output_dir = '/Users/dhanamon/LDAitkmitl/PyLDAVizOutput/th/'
# define an output directory to save an 'thai' pyLDAvis html file
th_pyLDAvis_output_file = 'th_5pdf_LDAvis_newmm_2n_postag_title_10n.html'

urls = [
        # 'https://elibrary.trf.or.th/fullP/SRI61X0602/SRI61X0602_full.pdf',
        'https://elibrary.trf.or.th/fullP/RDG6240001/RDG6240001_full.pdf',
        'https://elibrary.trf.or.th/fullP/RDG60T0025V01/unread_RDG60T0025V01_full.pdf',
        'https://elibrary.trf.or.th/fullP/SRI5851205/unread_SRI5851205_full.pdf',
        'https://elibrary.trf.or.th/fullP/RDG6210003/RDG6210003_full.pdf',
        # 'https://elibrary.trf.or.th/fullP/RDG6140033/RDG6140033_full.pdf',
        # 'https://elibrary.trf.or.th/fullP/RDG6140024/RDG6140024_full.pdf',
        # 'https://elibrary.trf.or.th/fullP/RDG6140012/RDG6140012_full.pdf',
        # 'https://elibrary.trf.or.th/fullP/RDG6140022/RDG6140022_full.pdf',
        # 'https://elibrary.trf.or.th/fullP/RDG6140023/RDG6140023_full.pdf',
        # 'https://elibrary.trf.or.th/fullP/RDG60H0018/RDG60H0018_full.pdf',
        # 'https://elibrary.trf.or.th/fullP/RDG6140010/RDG6140010_full.pdf',
        # 'https://elibrary.trf.or.th/fullP/PDG61M9005/PDG61M9005_full.pdf',
        # 'https://elibrary.trf.or.th/fullP/TRP62M0401/TRP62M0401_full.pdf',
        # 'https://elibrary.trf.or.th/fullP/SRI60M0415/SRI60M0415_full.pdf',
        # 'https://elibrary.trf.or.th/fullP/RDG61N0002/RDG61N0002_full.pdf',
        # 'https://elibrary.trf.or.th/fullP/RDG60N0007/RDG60N0007_full.pdf',
        # 'https://elibrary.trf.or.th/fullP/RDG60E0051/RDG60E0051_full.pdf',
        # 'https://elibrary.trf.or.th/fullP/RDG60N0004/RDG60N0004_full.pdf',
        # 'https://elibrary.trf.or.th/fullP/RDG60M0026/RDG60M0026_full.pdf',
        # 'https://elibrary.trf.or.th/fullP/RDG60N0020/RDG60N0020_full.pdf',
        # 'https://elibrary.trf.or.th/fullP/RDG6130005/RDG6130005_full.pdf'
        ]

titles = [
            # 'การศึกษาวิเคราะห์การทุจริตคอร์รัปชันของขบวนการเครือข่ายนายหน้าข้ามชาติในอุตสาหกรรมประมงต่อเนื่องของประเทศไทย',
            'นวัตกรรมเพื่อพัฒนาท้องถิ่นตามแนวทางปรัชญาของเศรษฐกิจพอเพียง: กรณีศึกษาองค์กรปกครองส่วนท้องถิ่นในจังหวัดนครสวรรค์และอุทัยธานี',
            'แผนงานวิจัย กลยุทธ์การเพิ่มขีดความสามารถการบริการการท่องเที่ยว เชิงส่งเสริมสุขภาพ ในจังหวัดภูเก็ต',
            'โครงการวิจัยเชงินโยบาย เพื่อยกระดับอุตสาหกรรมยานยนต์สีเขียวในประเทศไทย',
            'การศึกษาผลประโยชน์ทางธุรกิจที่เกิดจากการนำเศษพลอยมาใช้ประโยชน์ในเชิงพาณิชย์มากขึ้น',
            # 'การศึกษาประสบการณ์การเรียนรู้ของเยาวชนกลุ่มชาติพันธุ์ในการสร้างความรู้ด้านนิเวศวัฒนธรรม ',
            # 'การวิจัยและพัฒนากระบวนการผลิตและพัฒนาครูโดยบูรณาการแนวคิดจิตตปัญญาศึกษา ระบบพี่เลี้ยง และการวิจัยเป็นฐานของคณะครุศาสตร์ มหาวิทยาลัยราชภัฏ ปีที่ 2',
            # 'การพัฒนาชุดการเรียนรู้วิชาศิลปะในชั้นเรียนแบบเรียนรวมที่มีนักเรียนตาบอดระดับมัธยมศึกษาตอนปลายและการทดลองขยายผล',
            # 'การวิจัยและพัฒนากระบวนการผลิตและพัฒนาครูโดยบูรณาการแนวคิดจิตปัญญาศึกษา การเป็นพี่เลี้ยงและการวิจัยเป็นฐาน ภาคกลาง-ภาคตะวันตก  ปีที่ 2 ',
            # 'ระบบและกระบวนการผลิตและพัฒนาครูโดยใช้โครงงานฐานวิจัย ในพื้นที่ภาคใต้ ปีที่ 2',
            # 'การศึกษาประวัติศาสตร์สังคมพหุวัฒนธรรมจากตำนานประวัติศาสตร์ท้องถิ่นภาคใต้',
            # 'โครงการวิจัยและพัฒนาแนวทางการหนุนเสริมทางวิชาการเพื่อพัฒนากระบวนการผลิตและพัฒนาครูโดยบูรณาการแนวคิดจิตตปัญญาศึกษา ระบบพี่เลี้ยง และการวิจัยเป็นฐานของคณะครุศาสตร์ มหาวิทยาลัยราชภัฏ',
            # 'การหนุนเสริมศักยภาพนักวิจัยเยาวชนชาติพันธุ์รุ่นใหม่ 5 อำเภอชายแดนจังหวัดตาก ',
            # 'การศึกษาสถานการณ์ ศักยภาพและข้อจำกัดในการพัฒนาเศรษฐกิจฐานราก กรณีศึกษาภาคเหนือ',
            # 'ความเหลื่อมล้ำเชิงนโยบายด้านการเข้าถึงที่อยู่อาศัยของแรงงานนอกระบบในเมืองท่องเที่ยวกรณีศึกษา เมืองพัทยา จังหวัดชลบุรี',
            # 'แนวทางการฟื้นฟูทรัพยากรธรรมชาติและสิ่งแวดล้อมของชุมชนตำบลดงดำ อำเภอลี้ จังหวัดลำพูน',
            # 'การสร้างกระบวนการมีส่วนร่วมของชุมชนในการพัฒนาผลิตภัณฑ์ชาเชียงดาของศูนย์เรียนรู้เกษตรพอเพียง (หนองปลิง) เทศบาลตำบลวังผาง อำเภอเวียงหนองล่อง จังหวัดลำพูน',
            # 'รูปแบบกลไกการตลาดและการจัดการตลาดสินค้าอินทรีย์ที่เป็นธรรม ของเครือข่ายเกษตรอินทรีย์ จังหวัดยโสธร',
            # 'การพัฒนาหลักสูตรฝึกอบรมเพื่อพัฒนาขีดความสามารถชุมชนในการจัดการท่องเที่ยวโดยการมีส่วนร่วมของเครือข่ายชุมชนบ้านวังหาดและ กศน.อำเภอบ้านด่านลานหอย จังหวัดสุโขทัย',
            # 'ศูนย์วิจัยนวัตกรรมชุมชนภาคตะวันตกและการขยายผลการขับเคลื่อนงานวิจัยผ่านธุรกิจเพื่อสังคม',
            # 'การพัฒนาศูนย์วิจัยนวตกรรมชุมชนเพื่อขยายผลและยกระดับกระบวนการการจัดการงานวิจัยเพื่อท้องถิ่น ประเด็นการท่องเที่ยวโดยชุมชนภาคเหนือตอนล่าง',
            # 'โครงการศึกษาผลกระทบด้านขยะและน้ำเสียบริเวณพื้นที่เขตชายแดนบ้านคลองลึก อำเภออรัญประเทศ จังหวัดสระแก้ว    '
            ]


with open('json_request.json', 'r') as f:
    request_dict = json.load(f)

# print(json.dumps(request_dict, indent=4, sort_keys=True))

for request in request_dict:
    documents = request['documents']

project_id = request['project_id']
max_no_topic = request['max_no_topic']

print('========== Beginning file download with urllib2. ==========')
to_process_files = []
to_process_titles = []
error_doc_ids = []
counter = 0
for doc_id, document in documents.items():
    # print('document id: {0}'.format(doc_id))
    # print(document)

    url = document['url']
    file = Util.path_leaf(url)
    # print(file_)
    abs_file_path =  input_local_root + file
    # print(abs_file_path)

    if not os.path.isfile(abs_file_path):
        try:
            print('downloading file from this url: \"{0}\" with this file name : \"{1}\".'.format(url, file))
            urllib.request.urlretrieve(url, abs_file_path)
        except:
            print('An exception occurred when downloading a file from this url, \"{0}\"'.format(url))
            # Delete this document that cannot be downloaded at a specific index.
            del documents[doc_id]
            error_doc_ids.append(doc_id)
    else:
        print('-- This file, \"{0}\", already exists in: \"{1}\"! Therefore, this file will not be downloaded. --'.format(file, input_local_root))

    to_process_files.append(file)
    to_process_titles.append(document['title'])
    counter += 1

# print('========================')
# print(documents)

ldamodeling = LDAModeling()
ldamodeling.perform_topic_modeling(input_local_root, to_process_files, to_process_titles, converted_local_root,
                                   output_dir, pyLDAvis_output_file, th_output_dir, th_pyLDAvis_output_file,
                                   max_no_topic)


# max_no_topic = 10
#
# print('========== Beginning file download with urllib2. ==========')
# to_process_files = []
# abs_file_paths = []
# counter = 0
# #print(len(urls), len(titles))
# for url in urls:
#     file = Util.path_leaf(url)
#     # print(file_)
#     abs_file_path =  input_local_root + file
#     # print(abs_file_path)
#
#     if not os.path.isfile(abs_file_path):
#         try:
#             print('downloading file from this url: \"{0}\" with this file name : \"{1}\".'.format(url, file))
#             urllib.request.urlretrieve(url, abs_file_path)
#         except:
#             print('An exception occurred when downloading a file from this url, \"{0}\"'.format(url))
#             # Delete the title of a file that cannot be downloaded at a specific index.
#             # This is to keep two lists of abs_file_paths and titles consistent.
#
#             del titles[counter]
#
#     else:
#         print('-- This file, \"{0}\", already exists in: \"{1}\"! Therefore, this file will not be downloaded. --'.format(file, input_local_root))
#     to_process_files.append(file)
#     counter += 1
#
# ldamodeling = LDAModeling()
# ldamodeling.perform_topic_modeling(input_local_root, to_process_files, titles, converted_local_root,
#                                    output_dir, pyLDAvis_output_file, th_output_dir, th_pyLDAvis_output_file,
#                                    max_no_topic)


