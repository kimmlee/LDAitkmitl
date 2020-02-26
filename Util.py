import re
from PDFreader import pdfReader
from pdfminer.pdfparser import PDFSyntaxError
import docx2txt
import subprocess
import shlex

# This package is for parsing a file name
import ntpath

import os

"""
    A Utility (Util) class acts like a collection of methods to handles file management, loading and saving. 
    This class contains 4 static methods, including: 
        1) read_file()
        2) find_read_file()
        3) filter_file_to_read()
        4) path_leaf()
"""


class Util:
    """
        This class has not either instance or static attributes
    """

    """
      A static method, read_file(files), reads all files listed in a given List
    
    Parameters
    ----------
    files: a list of files in a string format, required
        
    Returns
    ----------
    data: a dictionary, in which a key is a file name and a value is a document text (raw text.)
        {
            "SRI61X0602_full": "การศึกษาวิเคราะห์การทุจริตคอร์รัปชันของขบวนการเครือข่ายนายหน้าข้ามชาติในอุตสาหกรรมประมงต่อเนื่องของประเทศไทย",
            "RDG6140010_full": "โครงการวิจัยและพัฒนาแนวทางการหนุนเสริมทางวิชาการเพื่อพัฒนากระบวนการผลิตและพัฒนาครูโดยบูรณาการแนวคิดจิตตปัญญาศึกษา",
            ...
        }
    """
    @staticmethod
    def read_file(local_path, files, converted_local_root):
        # create a dictionary, in which a key is a file name and a value is a document text (raw text.)
        data = {}
        # Read all given files in docx or readable pdf (readable means such a pdf file must be able to be read by pdfminer.)
        for file in files:
            data_file_text = ""
            file_path = local_path + file
            try:
                f_list = re.split("; |/|\\.", file_path)
                if file_path.endswith('.pdf'):
                    data_file_text = pdfReader.extract_pdf(file_path)
                elif file_path.endswith('.docx'):
                    data_file_text = docx2txt.process(file_path)

                # Add document text in a dictionary
                data[f_list[-2]] = [str(data_file_text)]
            except PDFSyntaxError as err:
                print('=======This file, {0}, is unreadable======='.format(file_path))
                print('Converting pdf by ghostscirpt')
                conv_file_path = converted_local_root + 'conv-' + Util.path_leaf(file_path)
                print(conv_file_path)
                if not os.path.isfile(conv_file_path):
                    call_with_args = "./ghostscript/convert_pdf2pdf_gs.sh '%s' '%s'" % (str(conv_file_path), str(file_path))
                    os.system(call_with_args)
                else:
                    print("This file, {0}, already exists and has previously been converted by ghostscript. So, it will not be converted again.".format(conv_file_path))

                try:
                    data_file_text = pdfReader.extract_pdf(conv_file_path)

                    # Add document text in a dictionary
                    data[f_list[-2]] = [str(data_file_text)]
                except Exception as inst:
                    print('Exception message: {0}'.format(inst))
            except:
                print("=======ERROR cannot find the below file in a given path=======")
                print(file_path, f_list)
                print('+++++++++++++++++++')

            # create rd_list for create training data
        #         data_file_text.close()
        return data

    # @staticmethod
    # def find_read_file(path, converted_local_root):
    #
    #     # Find all files in a given input path and list absolute paths to them in the variable files
    #     files = []
    #     for r, d, f in os.walk(path):
    #         for file in f:
    #             # Text file
    #             if 'news.txt' in file:
    #                 files.append(os.path.join(r, file))
    #             elif file.endswith('.pdf'):
    #                 print(file)
    #                 files.append(os.path.join(r, file))
    #             elif '.docx' in file:
    #                 print(file)
    #                 files.append(os.path.join(r, file))
    #
    #     data = self.read_file(files, converted_local_root)
    #     return data

    """
    A static method, filter_file_to_read(local_path, files), filters 'in' files in supported formats only (pdf, docx)
    
    Parameters
    ----------
    local_path: a local path that all files are storedt and located
    
    files: a list of files in a string format, required
        
    Returns
    ----------
    data: a dictionary, in which a key is a file name and a value is a document text (raw text.)
        {
            "SRI61X0602_full": "การศึกษาวิเคราะห์การทุจริตคอร์รัปชันของขบวนการเครือข่ายนายหน้าข้ามชาติในอุตสาหกรรมประมงต่อเนื่องของประเทศไทย",
            "RDG6140010_full": "โครงการวิจัยและพัฒนาแนวทางการหนุนเสริมทางวิชาการเพื่อพัฒนากระบวนการผลิตและพัฒนาครูโดยบูรณาการแนวคิดจิตตปัญญาศึกษา",
            ...
        }
    """
    @staticmethod
    def filter_file_to_read(local_path, files, converted_local_root):

        # Find all files in a given input path and list absolute paths to them in the variable files
        to_read_files = []
        for file in files:
            if file.endswith('.pdf'):
                print('-- To read file: \"{0}\". --'.format(file))
                to_read_files.append(file)
            elif file.endswith('.docx'):
                print('-- To read file: \"{0}\". --'.format(file))
                to_read_files.append(file)
            else:
                print(
                    '-- Only pdf and docx formats are supported. This file will be ignored due to not support types: \"{0}\". --'.format(
                        file))
        data = Util.read_file(local_path, to_read_files, converted_local_root)
        return data

    """
    A static method, path_leaf(path), returns a file name (leaf) from a given url or path
    
    Parameters
    ----------
    path: a path or url directing to a file 
        
    Returns
    ----------
    tail: a file name (leaf of path) if such a file exists; otherwise, a directory of one level above is returned 
    """
    @staticmethod
    def path_leaf(path):
        head, tail = ntpath.split(path)
        return tail or ntpath.basename(head)

    """
      A static method, path_dir(path), returns a directory where a given file is located from a given url or path

      Parameters
      ----------
      path: a path or url directing to a file 

      Returns
      ----------
      tail: a file name (leaf of path)
      """

    @staticmethod
    def path_dir(path):
        head, tail = ntpath.split(path)
        return head
