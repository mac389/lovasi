from Bio.Entrez import efetch, read

def fetch_abstract(pmid):
    handle = efetch(db='pubmed', id=pmid, retmode='xml')
    xml_data = read(handle)[0]
    try:
        article = xml_data['MedlineCitation']['Article']
        abstract = article['Abstract']['AbstractText'][0]
        return abstract
    except IndexError:
        return None
