"""
使用nlp_bert_document-segmentation_chinese-base模型，实现文本的语义级别分割
"""
from langchain_text_splitters import CharacterTextSplitter
import re
from typing import List
from modelscope.pipelines import pipeline
from base.config import Config

conf = Config()


class AliTextSplitter(CharacterTextSplitter):
    def __init__(self, pdf: bool = False, **kwargs):
        super().__init__(**kwargs)
        self.pdf = pdf

    def split_text(self, text: str) -> List[str]:
        # use_document_segmentation参数指定是否用语义切分文档，此处采取的文档语义分割模型为达摩院开源的nlp_bert_document-segmentation_chinese-base，论文见https://arxiv.org/abs/2107.09278
        # 如果使用模型进行文档语义切分，那么需要安装modelscope[nlp]：pip install "modelscope[nlp]" -f https://modelscope.oss-cn-beijing.aliyuncs.com/releases/repo.html
        # 考虑到使用了三个模型，可能对于低配置gpu不太友好，因此这里将模型load进cpu计算，有需要的话可以替换device为自己的显卡id
        if self.pdf:
            text = re.sub(r"\n{3,}", r"\n", text)
            text = re.sub(r'\s', " ", text)
            text = re.sub("\n\n", "", text)
        p = pipeline(
            task="document-segmentation",
            # model='/Users/chan/projects/Itcast_qa_system/rag_qa/models/nlp_bert_document-segmentation_chinese-base',
            model=conf.nlp_bert_doc_seg,
            device="cpu")
        result = p(documents=text)
        sent_list = [i for i in result["text"].split("\n\t") if i]
        return sent_list


if __name__ == '__main__':
    text = """
    美国国务院于当地时间2025年11月12日宣布，已组建名为“打击诈骗中心工作组”的特别行动组，以打击东南亚地区的加密货币投资诈骗活动。
　　据报道，美国财政部联合司法部等机构宣布组建Scam Center Strike Force，由哥伦比亚特区联邦检察官牵头，协同 DOJ、财政部、国务院及多部门打击通过加密交易实施的“杀猪盘”骗局，重点针对缅甸、柬埔寨、老挝、菲律宾等地的跨国网络。
　　美国哥伦比亚特区检察官珍妮·皮罗介绍，这一跨部门特别工作组旨在调查、打击和起诉最严重的网络诈骗犯罪。目前，美国希望成为全球加密货币行业的中心，因此加密货币的投资安全至关重要。该工作组将逮捕并起诉滥用民众信任的犯罪分子，以确保所有美国民众的投资安全。
　　据红星新闻报道，目前，该工作组已经在东南亚多地展开活动，行动地区包括缅甸、巴厘岛以及泰国。在缅甸，工作组查获用于连接互联网并实施诈骗和洗钱活动的卫星终端。在泰国，工作组还派遣了联邦调查局特工，与泰国皇家警察作战室特遣队合作，打击包括缅甸KK园区在内的诈骗窝点。
　　皮罗表示，该工作组从投入运作至今，已经查获了4亿美元的加密货币，并将没收另外8000万美元加密货币，返还给受害者。目前，美国希望成为全球加密货币行业的中心，因此加密货币的投资安全至关重要。该工作组将逮捕并起诉滥用民众信任的犯罪分子，以确保所有美国民众的投资安全。
　　据红星新闻报道，目前，该工作组已经在东南亚多地展开活动，行动地区包括缅甸、巴厘岛以及泰国。在缅甸，工作组查获用于连接互联网并实施诈骗和洗钱活动的卫星终端。在泰国，工作组还派遣了联邦调查局特工，与泰国皇家警察作战室特遣队合作，打击包括缅甸KK园区在内的诈骗窝点。
　　皮罗表示，该工作组从投入运作至今，已经查获了4亿美元的加密货币，并将没收另外8000万美元加密货币，返还给受害者。目前，美国希望成为全球加密货币行业的中心，因此加密货币的投资安全至关重要。该工作组将逮捕并起诉滥用民众信任的犯罪分子，以确保所有美国民众的投资安全。
　　据红星新闻报道，目前，该工作组已经在东南亚多地展开活动，行动地区包括缅甸、巴厘岛以及泰国。在缅甸，工作组查获用于连接互联网并实施诈骗和洗钱活动的卫星终端。在泰国，工作组还派遣了联邦调查局特工，与泰国皇家警察作战室特遣队合作，打击包括缅甸KK园区在内的诈骗窝点。
　　皮罗表示，该工作组从投入运作至今，已经查获了4亿美元的加密货币，并将没收另外8000万美元加密货币，返还给受害者。
    """
    model_split = AliTextSplitter()
    result = model_split.split_text(
        text=text, )
    for chunk in result:
        print(f"chunk size [{len(chunk)}]: {chunk}")
