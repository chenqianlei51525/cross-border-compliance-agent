'''
paddleocr：解析图片中的文字，也可以进行表格识别
rapidocr_paddle 和 rapidocr_onnxruntime 两种导入方式
主要区别在于它们所使用的推理引擎和硬件支持
选择哪种方式最合适取决于你的硬件环境和性能需求。
当你有 GPU 且追求速度时：使用 rapidocr_paddle。PaddlePaddle 原生支持在 GPU 上推理 PaddleOCR 模型，速度更快。
当只有 CPU 且需要高效推理时：使用 rapidocr_onnxruntime。它在 CPU 上进行了优化，资源占用较低.
'''


def get_ocr(use_cuda: bool = True):
    try:
        from rapidocr_paddle import RapidOCR
        '''
        det_use_cuda=True：启用检测模型的GPU加速。cls_use_cuda=True：启用分类模型的GPU加速。rec_use_cuda=True：启用识别模型的GPU加速。
        '''
        ocr = RapidOCR(det_use_cuda=use_cuda, cls_use_cuda=use_cuda, rec_use_cuda=use_cuda)
    except ImportError:
        try:
            from rapidocr_onnxruntime import RapidOCR
            ocr = RapidOCR()
        except ImportError:
            # rapidocr-onnxruntime 旧包不支持 Python 3.13，使用官方统一包并
            # 适配旧版 `(result, elapsed)` 返回格式，避免修改各文档加载器。
            from rapidocr import RapidOCR

            engine = RapidOCR()

            def ocr(image):
                output = engine(image)
                if output.boxes is None or output.txts is None:
                    return None, output.elapse_list
                result = [
                    [
                        box.tolist() if hasattr(box, "tolist") else box,
                        text,
                        float(score),
                    ]
                    for box, text, score in zip(
                        output.boxes, output.txts, output.scores
                    )
                ]
                return result, output.elapse_list
    return ocr


if __name__ == '__main__':
    model = get_ocr()
    result, data = model("/Users/chan/projects/Itcast_qa_system/rag_qa/data/samples/ocr_04.png")
    for each in result:
        print(each)
