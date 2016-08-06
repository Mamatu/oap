package ogla.core;

/**
 * Thanks this class you can create help sites of your plugin. This section will be displayed
 * in help section of application.
 */
public interface Help {

    /**
     * Get main chapter.
     * @return main chapter
     */
    public Chapter getRootChapter();

    /**
     * Interfacce which represent document.
     */
    public interface Document {

        /**
         * Get title of this document.
         * @return title
         */
        public String getTitle();

        /**
         * Get content of this document.
         * @return content
         */
        public String getContent();
    }

    public interface Chapter {

        /**
         * Get title of this document.
         * @return title
         */
        public String getTitle();

        /**
         * Get documents which are attached to this chapter.
         * @return array of documents
         */
        public Document[] getDocuments();

        /**
         * Get subchapters.
         * @return array of subchapters which are attached to this chapter.
         */
        public Chapter[] getSubChapters();
    }
}
