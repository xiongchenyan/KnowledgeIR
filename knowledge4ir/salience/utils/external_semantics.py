"""
11/15/2017 Chenyan

models to use external semantics in entity salience modeling

external semantics now:
    description in ExtData
    will have NLSS and RDF in the future

model:
    get an additional entity embedding from the ext semantics
    embedding can be:
        average of desp's first 10 words
        weighted average of the first 10 words
        desp RNN
        desp's senence's RNN with attention
        desp CNN
    to use the additional embedding
        add | concatenate with e's learned embedding | caocatenate + projection to a new embedding

    cost:
        for each doc, need get all its e's external semantics

"""