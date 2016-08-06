/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package ogla.core.ui;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import ogla.core.util.ArgumentType;
import ogla.core.util.ArgumentsUtils;

/**
 *
 * @author mmatula
 */
public class OglaObject implements Documentation {

    private ObjectInfo objectInfo = null;

    @Override
    public String getText() {
        return this.objectInfo.getText();
    }

    @Override
    public String getTextWithChildren() {
        return this.objectInfo.getTextWithChildren();
    }

    public OglaObject getRoot() {
        return this.root;
    }

    private OglaObject root = null;
    private List<OglaObject> roots = null;

    public OglaObject(String name, OglaObject root, List<OglaObject> roots) {
        this.name = name;
        objectInfo = this.new ObjectInfo();
        this.root = root;
        this.roots = roots;
    }

    private HashMap<String, OglaMethod> namesMethods = new HashMap<String, OglaMethod>();
    private HashMap<OglaMethod, Object> oglaUserData = new HashMap<OglaMethod, Object>();
    private List<OglaMethod> methods = new ArrayList<OglaMethod>();

    public boolean isMethod(String name, ArgumentType[] inargs) {
        for (OglaMethod method : namesMethods.values()) {
            if (method.getName().equals(name) && method.equalsArgumentType(inargs)) {
                return true;
            }
        }
        return false;
    }

    public void addMethod(OglaMethod method, Object userData) {
        namesMethods.put(method.getName(), method);
        methods.add(method);
        oglaUserData.put(method, userData);
        this.objectInfo.addChild(method);
    }

    public void addMethod(OglaMethod method) {
        this.addMethod(method, null);
    }

    private String name = null;
    private String[] names = null;

    private static void getNames(List<String> names, OglaObject oglaObject) {
        if (oglaObject.getRoot() == null) {
            names.add(oglaObject.name);
        } else {
            getNames(names, oglaObject.getRoot());
            names.add(oglaObject.name);
        }
    }

    public static OglaObject getFirstRoot(OglaObject oglaObject) {
        OglaObject root = oglaObject;
        while (root.getRoot() != null) {
            root = root.getRoot();
        }
        return root;
    }

    private static String[] getNames(OglaObject oglaObject, String... extraNames) {
        List<String> names = new ArrayList<String>();
        getNames(names, oglaObject);
        String[] array = new String[names.size()];
        for (String n : extraNames) {
            names.add(n);
        }
        return names.toArray(array);
    }

    public OglaMethod getMethod(String name, ArgumentType[] inargs) {
        for (OglaMethod method : namesMethods.values()) {
            if (method.getName().equals(name) && method.equalsArgumentType(inargs)) {
                return method;
            }
        }
        return null;
    }

    public Object getUserData(OglaMethod oglaMethod) {
        return this.oglaUserData.get(oglaMethod);
    }

    public Object invokeMethod(String arg0, Object arg1) {
        ArgumentType[] argumentTypes = ArgumentsUtils.getArguments(arg1);
        OglaMethod oglaMethod = this.getMethod(name, argumentTypes);
        Object userData = this.getUserData(oglaMethod);
        return oglaMethod.invoke(arg1, root, userData);
    }

    public OglaMethod getMethod(int index) {
        return this.methods.get(index);
    }

    public int getMethodsCount() {
        return this.methods.size();
    }

    private class ObjectInfo extends DocumentationImpl {

        public String getDescription() {
            return "";
        }

        public String getInfo() {
            StringBuilder builder = new StringBuilder();
            for (int fa = 0; fa < OglaObject.this.roots.size() - 1; fa++) {
                builder.append(OglaObject.this.roots.get(fa).getName());
                builder.append("::");
            }
            builder.append(OglaObject.this.getName());
            return builder.toString();
        }
    }

    public String getName() {
        return this.name;
    }

    public String getFullName() {
        return this.objectInfo.getInfo();
    }

    public List<String> getNames(List<String> list) {
        for (OglaObject root : this.roots) {
            list.add(root.getName());
        }
        list.add(this.getName());
        return list;
    }
}
