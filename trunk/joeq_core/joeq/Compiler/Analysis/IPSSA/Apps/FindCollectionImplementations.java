package Compil3r.Analysis.IPSSA.Apps;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.Arrays;
import java.util.Collection;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashSet;
import java.util.Iterator;
import java.util.LinkedList;
import java.util.List;
import java.util.Set;
import java.util.StringTokenizer;

import Bootstrap.PrimordialClassLoader;
import Clazz.jq_Class;
import Clazz.jq_Method;
import Clazz.jq_Type;
import Compil3r.Quad.CachedCallGraph;
import Compil3r.Quad.CallGraph;
import Compil3r.Quad.RootedCHACallGraph;
import Main.HostedVM;
import Util.Assert;
import Util.Collections.AppendIterator;

class ClassHierarchy {
    protected class ClassHieraryNode {
        Set              _children  = new HashSet();
        jq_Class         _class     = null;
        ClassHieraryNode _parent    = null;
        
        ClassHieraryNode(jq_Class c){
            this._class = c;
        }
        private void addChild(ClassHieraryNode n) {
            //if(!_children.contains(n)) {
                // adding twice shouldn't matter
                _children.add(n);
            //}
        }
        public int getChildCount() {
            return _children.size();
        }
        public Iterator getChildIterator() {
            return _children.iterator();
        }
        public jq_Class getClazz() {
            return _class;
        }
        public void reset() {
            this._parent = null;
            _children = new HashSet();            
        }
        public void setRoot(ClassHieraryNode n) {
            //System.err.println("Connecting " + this + " and " + n);
            this._parent = n;
            n.addChild(this);
        }
        public String toLongString() {
            return _class.getJDKDesc();
        }
        public String toString(){
            return _class.toString();                
        }
        public List getChilden() {
            return Arrays.asList(_children.toArray());
        }
    }
    Set _nodes = new HashSet();     
    ClassHieraryNode _root  = null;
    
    ClassHierarchy(ClassHieraryNode root){
        this._root = root;
        add(_root);
    }

    ClassHierarchy(jq_Class root){
        this._root = new ClassHieraryNode(root);
        add(_root);
    }
    
    public ClassHierarchy(jq_Class root, Collection c) {
        this._root = new ClassHieraryNode(root);
        Assert._assert(_root != null);

        add(_root);
        
        for(Iterator iter = c.iterator(); iter.hasNext();) {
            jq_Class c2 = (jq_Class)iter.next();
            
            add(c2);
        }
    }
    
    private void add(ClassHieraryNode node) {
        _nodes.add(node);
    }

    void add(jq_Class c) {
        if(!hasClass(c)) {
            _nodes.add(new ClassHieraryNode(c));
        }
    }

    private ClassHieraryNode getClassNode(jq_Class c) {
        // lame linear search
        for(Iterator iter = _nodes.iterator(); iter.hasNext();) {
            ClassHieraryNode node = (ClassHieraryNode)iter.next();
            
            if(node.getClazz() == c) {
                return node;
            }
        }
        
        return null;
    }
    
    boolean hasClass(jq_Class c) {
        return getClassNode(c) != null;
    }
    
    public void makeHierarchy() {
        if(_nodes.size() <= 1) return;
        Assert._assert(_root != null, "Root is not set in the beginning of makeHierarchy");
        // clear potential all data
        resetNodes();
        // use the nodes currently in the set and reset the links
        for(Iterator iter = _nodes.iterator(); iter.hasNext();) {
            ClassHieraryNode node = (ClassHieraryNode)iter.next();
            jq_Class c = node.getClazz();
            
            do {
                if(c instanceof jq_Class && ((jq_Class)c).isInterface()){
                    //System.err.println("Reached interface: " + c);
                }
                // directly supports this interface
                if(c.getDeclaredInterface(_root.getClazz().getDesc()) != null){
                    // termination condition
                    //System.err.println("Reached root: " + c);
                    if(node != _root){
                        node.setRoot(_root);
                        Assert._assert(_root.getChildCount() > 0);
                    }
                    break;
                }
                jq_Class superClass = (jq_Class)c.getSuperclass();
                ClassHieraryNode n = getClassNode(superClass);
                if(n != null) {
                    // found the most direct link -- make the connection
                    node.setRoot(n);
                }
                if(superClass == c){
                    break; // self-recursion
                }
                c = superClass;
            } while(c != null);            
        }
        Assert._assert(_root != null, "Root is not set at the end of makeHierarchy");
        Assert._assert(_root.getChildCount() > 0, "Root is not connected to any children");
    }
    
    public void printHierarchy() {
        if(size() <= 0) return;
        Assert._assert(_root != null);
        
        System.out.println("Printing a hierarchy of size " + size() + " rooted at " + _root);
        printHierarchyAux(_root, "");
    }
    
    private int size() {
        return _nodes.size() - 1;
    }

    /**
     * Compares class names.
     * */
    public class ClassComparator implements Comparator {
        public int compare(Object arg0, Object arg1) {            
            return arg0.toString().toLowerCase().compareTo(arg1.toString().toLowerCase());
        }
    }
    
    private void printHierarchyAux(ClassHieraryNode node, String string) {        
        System.out.print(string + node.toString());
        System.out.println(node.getChildCount() == 0 ? "" : (" " + node.getChildCount()));
        List children = node.getChilden();
        Comparator comparator = new ClassComparator();        
        Collections.sort(children, comparator);
        for(Iterator iter = children.iterator(); iter.hasNext();) {
            ClassHieraryNode child = (ClassHieraryNode)iter.next();

            Assert._assert(child != node, "Child: " + child + " is the same as " + node);            
            printHierarchyAux(child, string + "\t");
        }
    }

    private void resetNodes() {
        Assert._assert(_root != null, "Root is not set in the beginning of resetNodes");
        for(Iterator iter = _nodes.iterator(); iter.hasNext();) {
            ClassHieraryNode node = (ClassHieraryNode)iter.next();
            node.reset();
        }
        Assert._assert(_root != null, "Root is not set at the end of resetNodes");
    }
}

public class FindCollectionImplementations {    
    private static CallGraph _cg;
    
    static jq_Class _collectionClass  = null;
    static jq_Class _iteratorClass    = null;
    boolean FILTER = false;

    static final String COLLECTION_SIGNATURE = "Ljava.util.Collection;";
    static final String ITERATOR_SIGNATURE   = "Ljava.util.Iterator;";    
    
    public static void main(String[] args) {
        HostedVM.initialize();
        
        Iterator i = null;
        for (int x=0; x<args.length; ++x) {
            if (args[x].equals("-file")) {
                try {
                    BufferedReader br = new BufferedReader(new FileReader(args[++x]));
                    LinkedList list = new LinkedList();
                    for (;;) {
                        String s = br.readLine();
                        if (s == null) break;
                        if (s.length() == 0) continue;
                        if (s.startsWith("%")) continue;
                        if (s.startsWith("#")) continue;
                        list.add(s);
                    }
                    i = new AppendIterator(list.iterator(), i);
                }catch(IOException e) {
                    e.printStackTrace();
                    System.exit(2);
                }
                
            } else
            if (args[x].endsWith("*")) {
                i = new AppendIterator(PrimordialClassLoader.loader.listPackage(args[x].substring(0, args[x].length()-1)), i);
            } else 
            if(args[x].charAt(0) == '-'){
                System.exit(2);                    
            }else {
                String classname = args[x];
                i = new AppendIterator(Collections.singleton(classname).iterator(), i);
            }
        }

        FindCollectionImplementations finder = new FindCollectionImplementations(i);
        finder.run();
    }
    private Set _classes;
    private Set _collections;
    private Set _iterators;
    
    
    public FindCollectionImplementations(Iterator i) {
        Collection roots = new LinkedList();
        Collection root_classes = new LinkedList();
        while(i.hasNext()) {
            jq_Class c = (jq_Class) jq_Type.parseType((String)i.next());
            c.load();
            root_classes.add(c);

            roots.addAll(Arrays.asList(c.getDeclaredStaticMethods()));
        }
        
        //System.out.println("Classes: " + classes);
        System.out.println("Roots: " + roots);
        
        System.out.print("Building call graph...");
        long time = System.currentTimeMillis();
        _cg = new RootedCHACallGraph();
        _cg.setRoots(roots);
        //_cg = new CachedCallGraph(_cg);
        
        time = System.currentTimeMillis() - time;
        System.out.println("done. ("+(time/1000.)+" seconds)");
        _classes = getClasses(_cg.getAllMethods());
        if(FILTER) _classes = filter(_classes, root_classes);
        
        if(FILTER){
    	    System.out.println("Considering classes: " + _classes);
    	}
        
        _collections = new HashSet();
        _iterators   = new HashSet();
        
        _collectionClass  = (jq_Class)jq_Type.parseType(COLLECTION_SIGNATURE);
        _iteratorClass    = (jq_Class)jq_Type.parseType(ITERATOR_SIGNATURE);  
        _collectionClass.load();
        _iteratorClass.load();

        Assert._assert(_collectionClass != null);
        Assert._assert(_iteratorClass  != null);
    }
    
    private Set filter(Set classes, Collection roots) {
        Set prefixes = new HashSet();
        for(Iterator iter = roots.iterator(); iter.hasNext();) {
            jq_Class root  = (jq_Class)iter.next();
            StringTokenizer t = new StringTokenizer(root.getJDKDesc(), ".");
            String prefix = t.nextToken();
            prefixes.add(prefix);
        }
        System.out.println("Recognized prefixes: " + prefixes);
        
        Set result = new HashSet();
        for(Iterator iter = classes.iterator(); iter.hasNext();) {
            jq_Class c          = (jq_Class)iter.next();
            StringTokenizer t   = new StringTokenizer(c.getJDKDesc(), ".");
            String prefix       = t.nextToken();
            
            if(prefixes.contains(prefix)) {
                result.add(c);
            }
        }
        
        return result;
    }

    private void findCollections() {      
        for(Iterator iter = _classes.iterator(); iter.hasNext(); ) {
            jq_Class c = (jq_Class)iter.next();
            
            if(c.getDeclaredInterface(_collectionClass.getDesc()) != null) {
                _collections.add(c);
            }
        }        
    }
    private void findIterators() {        
        for(Iterator iter = _classes.iterator(); iter.hasNext(); ) {
            jq_Class c = (jq_Class)iter.next();
            
            if(c.getDeclaredInterface(_iteratorClass.getDesc()) != null) {
                _iterators.add(c);
            }
        }        
    }

    private Set getClasses(Collection collection) {
        HashSet result = new HashSet(); 
        for(Iterator iter = collection.iterator(); iter.hasNext(); ) {
            jq_Method method = (jq_Method)iter.next();
            //System.err.println("Saw " + method);
         
            jq_Class c = method.getDeclaringClass();
            if(c != null) {
                result.add(c);
            }
        }
        
        return result;
    }

    private void printCollection(Collection collection) {
        Iterator iter = collection.iterator();
        while(iter.hasNext()) {
            jq_Class c = (jq_Class)iter.next();
            
            System.out.println("\t" + c);
        }
    }
    
    private void reportStats() {
        System.out.println("Found " + _collections.size() + " collections:");
        //printCollection(_collections);
        ClassHierarchy h = new ClassHierarchy(_collectionClass, _collections);
        h.makeHierarchy();
        h.printHierarchy();
        
        System.out.println("Found " + _iterators.size() + " iterators");
        //printCollection(_iterators);
        h = new ClassHierarchy(_iteratorClass, _iterators);
        h.makeHierarchy();
        h.printHierarchy();

	System.out.println("Found " + _collections.size() + " collections, " + _iterators.size() + " iterators");
    }
    
    protected void run() {        
        System.out.println("Looking for subclasses of " + _collectionClass + " and " + _iteratorClass);
        
        findCollections();
        findIterators();        
        
        reportStats();
    }
}
