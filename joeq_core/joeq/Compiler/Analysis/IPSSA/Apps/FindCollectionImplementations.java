package Compil3r.Analysis.IPSSA.Apps;

import java.util.Arrays;
import java.util.Collection;
import java.util.HashSet;
import java.util.Iterator;
import java.util.LinkedList;
import java.util.Set;

import Clazz.jq_Class;
import Clazz.jq_Method;
import Clazz.jq_Type;
import Compil3r.Quad.CachedCallGraph;
import Compil3r.Quad.CallGraph;
import Compil3r.Quad.RootedCHACallGraph;
import Main.HostedVM;
import Util.Assert;

public class FindCollectionImplementations {    
    Collection roots;
    private static CallGraph _cg;
    private Collection _classes;
    private Collection _collections;
    private Collection _iterators;

    static final String COLLECTION_SIGNATURE = "Ljava.util.Collection;";
    static final String ITERATOR_SIGNATURE   = "Ljava.util.Iterator;";    
    
    static jq_Class _collectionClass  = null;
    static jq_Class _iteratorClass    = null;
    
    public FindCollectionImplementations(String startClass) {
        jq_Class c = (jq_Class) jq_Type.parseType(startClass);
        c.prepare();
        
        System.out.print("Building call graph...");
        long time = System.currentTimeMillis();
        _cg = new RootedCHACallGraph();
        _cg = new CachedCallGraph(_cg);
        Collection roots = Arrays.asList(c.getDeclaredStaticMethods());
        _cg.setRoots(roots);
        time = System.currentTimeMillis() - time;
        System.out.println("done. ("+(time/1000.)+" seconds)");
        _classes = getClasses(_cg.getAllMethods());
        
        _collections = new LinkedList();
        _iterators   = new LinkedList();
        
        _collectionClass  = (jq_Class)jq_Type.parseType(COLLECTION_SIGNATURE);
        _iteratorClass    = (jq_Class)jq_Type.parseType(ITERATOR_SIGNATURE);  

        Assert._assert(_collectionClass != null);
        Assert._assert(_iteratorClass  != null);
    }
    
    public static void main(String[] args) {
        HostedVM.initialize();

        FindCollectionImplementations finder = new FindCollectionImplementations(args[0]);
        finder.run();
    }
    
    protected void run() {        
        System.err.println("Looking for subclasses of " + _collectionClass + " and " + _iteratorClass);
        
        findCollections();
        findIterators();        
        
        reportStats();
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
    }

    private void printCollection(Collection collection) {
        Iterator iter = collection.iterator();
        while(iter.hasNext()) {
            jq_Class c = (jq_Class)iter.next();
            
            System.out.println("\t" + c);
        }
    }

    private Collection getClasses(Collection collection) {
        LinkedList result = new LinkedList(); 
        for(Iterator iter = collection.iterator(); iter.hasNext(); ) {
            jq_Method method = (jq_Method)iter.next();
         
            jq_Class c = method.getDeclaringClass();
            if(c != null) {
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
}

class ClassHierarchy {
    protected class ClassHieraryNode {
        ClassHieraryNode _parent    = null;
        jq_Class         _class     = null;
        Set              _children  = new HashSet();
        
        ClassHieraryNode(jq_Class c){
            this._class = c;
        }
        public jq_Class getClazz() {
            return _class;
        }
        public void setRoot(ClassHieraryNode n) {
            //System.err.println("Connecting " + this + " and " + n);
            this._parent = n;
            n.addChild(this);
        }
        private void addChild(ClassHieraryNode n) {
            _children.add(n);
        }
        public void reset() {
            this._parent = null;
            _children = new HashSet();            
        }
        public Iterator getChildIterator() {
            return _children.iterator();
        }
        public String toString(){
            return _class.toString();                
        }
        public int getChildCount() {
            return _children.size();
        }
    }
    
    ClassHieraryNode _root  = null;
    Set              _nodes = new HashSet();  
    
    ClassHierarchy(ClassHieraryNode root){
        this._root = root;
        add(_root);
    }
    
    private void add(ClassHieraryNode node) {
        _nodes.add(node);
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

    void add(jq_Class c) {
        _nodes.add(new ClassHieraryNode(c));
    }
    
    boolean hasClass(jq_Class c) {
        return getClassNode(c) != null;
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
    
    public void makeHierarchy() {
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
        Assert._assert(_root != null);
        
        System.out.println("Printing a hierarchy rooted at " + _root);
        printHierarchyAux(_root, "");
    }
    
    private void printHierarchyAux(ClassHieraryNode node, String string) {        
        System.out.print(string + node);
        System.out.println(node.getChildCount() == 0 ? "" : (" " + node.getChildCount()));
        for(Iterator iter = node.getChildIterator(); iter.hasNext();) {
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
