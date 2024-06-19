from src.fetch_web_content import WebContentFetcher
from src.retrieval import EmbeddingRetriever
from src.llm_answer import GPTAnswer



def get_company_desc(comp, year):
    query = f"{comp} technologies business operations before:{year} after:{str(int(year)-10)}" # use the last 10 years of public info. Example: for 2000, use 1990-2000 web search results. Works quite bad with year < 1990.
    output_format = "1. Technology #1: description, risk of disruption, potential disruptors. Do this for every technology." # User can specify output format

    # Fetch web content based on the query
    web_contents_fetcher = WebContentFetcher(query)
    web_contents, serper_response = web_contents_fetcher.fetch()
    # print(web_contents)
    # print(serper_response)
    # Retrieve relevant documents using embeddings
    content_processor = GPTAnswer()
    try:
        retriever = EmbeddingRetriever()
        relevant_docs_list = retriever.retrieve_embeddings(web_contents, serper_response['links'], query)
        formatted_relevant_docs = content_processor._format_reference(relevant_docs_list, serper_response['links'])
    except:
        formatted_relevant_docs = ""
    # print(formatted_relevant_docs)
    # Generate answer from ChatOpenAI
    ai_message_obj = content_processor.get_answer(query, formatted_relevant_docs, output_format, comp, year)
    # answer = ai_message_obj
    answer = ai_message_obj.content + '\n'
    return answer


if __name__ == '__main__':
    print(get_company_desc('NVIDIA INC', '2012'))
