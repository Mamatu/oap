package ogla.core.util;

import ogla.core.Help.Chapter;
import ogla.core.Help.Document;
import java.util.ArrayList;
import java.util.List;

public class DefaultChapterImpl implements Chapter {

    public String title;
    public List<Document> documents = new ArrayList<Document>();
    public List<Chapter> subchapters = new ArrayList<Chapter>();

    public DefaultChapterImpl(String title) {
        this.title = title;
    }

    public String getTitle() {
        return title;
    }

    public Document[] getDocuments() {
        Document[] docs = new Document[documents.size()];
        return documents.toArray(docs);
    }

    public Chapter[] getSubChapters() {
        Chapter[] sub = new Chapter[subchapters.size()];
        return subchapters.toArray(sub);
    }
}
