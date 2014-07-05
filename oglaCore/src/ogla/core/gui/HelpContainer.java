package ogla.core.gui;

import ogla.core.Help;
import java.util.HashMap;
import java.util.Map;
import javax.swing.tree.DefaultMutableTreeNode;
import javax.swing.tree.DefaultTreeModel;
import javax.swing.tree.TreeModel;
import ogla.core.Help;

/**
 *
 * @author marcin
 */
public class HelpContainer {

    public class LabRoot {

        protected String label;

        public LabRoot(String label) {
            this.label = label;
        }

        @Override
        public String toString() {
            return label;
        }
    }

    public class LabChapter extends LabRoot {

        public Help.Chapter chapter;
        public DefaultMutableTreeNode node = null;

        public void createNode() {
            node = new DefaultMutableTreeNode(this);
        }

        public LabChapter(Help.Chapter chapter) {
            super(chapter.getTitle());
            this.chapter = chapter;
        }
    }

    public class LabDocument extends LabRoot {

        public Help.Document document;
        public DefaultMutableTreeNode node = null;

        public void createNode() {
            node = new DefaultMutableTreeNode(this);
        }

        public LabDocument(Help.Document document) {
            super(document.getTitle());
            this.document = document;
        }
    }
    protected DefaultMutableTreeNode root = new DefaultMutableTreeNode(new LabRoot("Help"));
    protected DefaultTreeModel treeModel = new DefaultTreeModel(root);
    private Map<Help.Chapter, DefaultMutableTreeNode> map = new HashMap<Help.Chapter, DefaultMutableTreeNode>();

    private void addDocument(Help.Document[] documents, DefaultMutableTreeNode pnode) {
        if (documents == null) {
            return;
        }
        for (Help.Document document : documents) {
            LabDocument labDocument = new LabDocument(document);
            labDocument.createNode();
            pnode.add(labDocument.node);
        }
    }

    private void createSubTree(LabChapter labChapter, DefaultMutableTreeNode pnode) {
        pnode.add(labChapter.node);
        if (labChapter.chapter.getSubChapters() != null) {
            for (Help.Chapter chapter : labChapter.chapter.getSubChapters()) {
                LabChapter labChapter1 = new LabChapter(chapter);
                labChapter1.createNode();
                createSubTree(labChapter1, labChapter.node);
            }
        }
        addDocument(labChapter.chapter.getDocuments(), labChapter.node);
    }

    public boolean add(Help help) {
        LabChapter rootChapter = new LabChapter(help.getRootChapter());
        rootChapter.createNode();
        createSubTree(rootChapter, root);
        map.put(rootChapter.chapter, rootChapter.node);
        return true;
    }

    public boolean remove(Help help) {
        DefaultMutableTreeNode node = map.get(help.getRootChapter());
        root.remove(node);
        return true;
    }

    public TreeModel getModel() {
        return treeModel;
    }
}
