from PDFreader.pdfReader import extract_pdf
import docx2txt
from pdfminer.pdfparser import PDFSyntaxError
import re
import os
import natsort 
import ntpath


class Util:
    @staticmethod
    def read_file(files, converted_local_root, unreadable_docs):
        data_file =[]
        data={}
        # converted_local_root = '/Users/dhanamon/LDAitkmitl/document/TestDownloadFiles/converted/'
        for f in files:
            # print(f)
            data_file_text = ""
            f_list = re.split("; |/|\\.", f)
            try:
                if f.endswith('.pdf'):
                    # print("pdf",f)
                    data_file_text = extract_pdf(f)
                elif f.endswith('.docx'):
                    # print("docx",f)
                    data_file_text = docx2txt.process(f)

                # Add document text in a dictionary
                data[f_list[-2]] = [str(data_file_text)]
            except PDFSyntaxError as err:
                print('=======This file, {0}, is unreadable======='.format(f))
                print('Converting pdf by ghostscirpt')
                conv_file_path = converted_local_root + 'conv-' + Util.path_leaf(f)
                print(conv_file_path)
                if not os.path.isfile(conv_file_path):
                    call_with_args = "./ghostscript/convert_pdf2pdf_gs.sh '%s' '%s'" % (str(conv_file_path), str(f))
                    os.system(call_with_args)
                else:
                    print("This file, {0}, already exists and has previously been converted by ghostscript. So, it will not be converted again.".format(conv_file_path))

                try:
                    data_file_text = extract_pdf(conv_file_path)

                    # Add document text in a dictionary
                    data[f_list[-2]] = [str(data_file_text)]
                except Exception as inst:
                    print('Exception message: {0}'.format(inst))
                    unreadable_docs.append(f_list[-2])
            except:
                print("=======ERROR cannot find the below file in a given path=======")
                print(f, f_list)
                unreadable_docs.append(f_list[-2])
                print('+++++++++++++++++++')
        return data, unreadable_docs
    @staticmethod
    def find_read_file(doc_path_dict, converted_local_root, unreadable_docs):
        """
        Finding file in input path file and keep in dictionary
        """
        files = []
        for proj_id in doc_path_dict.keys():
            files.append(doc_path_dict[proj_id])
        files = natsort.natsorted(files,reverse=True)
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
        data, unreadable_docs = Util.read_file(to_read_files, converted_local_root, unreadable_docs)
        
        # print(files)
        #Find file in folder path and extract file to text
        # data = self.read_file(files)
        return data, unreadable_docs
    @staticmethod
    def path_leaf(path):
        head, tail = ntpath.split(path)
        return tail or ntpath.basename(head)