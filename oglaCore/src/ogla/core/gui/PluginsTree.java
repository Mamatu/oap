package ogla.core.gui;

import java.util.LinkedHashSet;
import java.util.Set;
import javax.swing.tree.DefaultMutableTreeNode;
import javax.swing.tree.DefaultTreeModel;
import javax.swing.tree.TreeNode;
import javax.swing.tree.TreePath;

public class PluginsTree {

    protected DefaultTreeModel defaultTreeModel = null;
    protected DefaultMutableTreeNode root = new DefaultMutableTreeNode("Plugins");
    protected DefaultMutableTreeNode others = new DefaultMutableTreeNode("others");
    protected DefaultMutableTreeNode sections = new DefaultMutableTreeNode("sections");
    protected Set<DefaultMutableTreeNode> pluginsNode = new LinkedHashSet<DefaultMutableTreeNode>();
    protected Set<DefaultMutableTreeNode> sectionsNode = new LinkedHashSet<DefaultMutableTreeNode>();

    private void addToSection(String section, Plugin plugin, DefaultMutableTreeNode pluginNode) {
        DefaultMutableTreeNode sectionNode = null;
        if (section != null && !section.equals("")) {
            for (DefaultMutableTreeNode node : sectionsNode) {
                String sectionOfNode = (String) node.getUserObject();
                if (sectionOfNode.equals(section)) {
                    sectionNode = node;
                    break;
                }
            }
        }
        if (sectionNode == null) {
            if (plugin.getSection() == null || plugin.getSection().equals("")) {
                sectionNode = others;
            } else {
                DefaultMutableTreeNode newSection = new DefaultMutableTreeNode(plugin.getSection());
                sectionNode = newSection;
                sectionsNode.add(newSection);
                sections.add(newSection);
                defaultTreeModel.insertNodeInto(newSection, sections, sections.getChildCount() - 1);
            }
        }
        defaultTreeModel.reload();
        sectionNode.add(pluginNode);
    }

    public PluginsTree() {
        defaultTreeModel = new DefaultTreeModel(root);
        root.add(others);
        root.add(sections);
        sectionsNode.add(sections);
        sectionsNode.add(others);
    }

    public void remove(Plugin plugin) {
        defaultTreeModel.removeNodeFromParent(plugin.getTreeNode());
    }

    public Plugin get(TreePath path) {
        for (DefaultMutableTreeNode node : pluginsNode) {
            if ((TreeNode) node == (TreeNode) path.getLastPathComponent()) {
                return (Plugin) node.getUserObject();
            }
        }
        return null;
    }

    public void add(Plugin plugin) {
        pluginsNode.add(plugin.getTreeNode());
        DefaultMutableTreeNode dependencies = new DefaultMutableTreeNode("dependencies");
        DefaultMutableTreeNode state = new DefaultMutableTreeNode("state");
        plugin.getTreeNode().add(dependencies);
        if (plugin.getDependecies() != null) {
            for (String d : plugin.getDependecies()) {
                dependencies.add(new DefaultMutableTreeNode(d));
            }
        }
        addToSection(plugin.getSection(), plugin, plugin.getTreeNode());
    }
}
