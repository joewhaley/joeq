package Util;

import java.util.Collection;

/**
 * @author John Whaley
 * @version $Id$
 */
public abstract class BDDFactory {

    public static final BDDOp and   = new BDDOp(1, "and");
    public static final BDDOp xor   = new BDDOp(2, "xor");
    public static final BDDOp or    = new BDDOp(3, "or");
    public static final BDDOp nand  = new BDDOp(4, "nand");
    public static final BDDOp nor   = new BDDOp(5, "nor");
    public static final BDDOp imp   = new BDDOp(6, "imp");
    public static final BDDOp biimp = new BDDOp(7, "biimp");
    public static final BDDOp diff  = new BDDOp(8, "diff");
    public static final BDDOp less  = new BDDOp(9, "less");
    public static final BDDOp invimp = new BDDOp(10, "invimp");

    public static class BDDOp {
        final int id; final String name;
        private BDDOp(int id, String name) {
            this.id = id;
            this.name = name;
        }
        public String toString() {
            return name;
        }
    }
    
    /**
     * Get the constant false BDD.
     * 
     * Compare to bdd_false.
     */
    public abstract BDD zero();
    
    /**
     * Get the constant true BDD.
     * 
     * Compare to bdd_true.
     */
    public abstract BDD one();
    
    /**
     * Build a cube from an array of variables.
     * 
     * Compare to bdd_buildcube.
     */
    public abstract BDD buildCube(int value, int width, Collection/*BDD*/ var);
    
    /**
     * Build a cube from an array of variables.
     * 
     * Compare to bdd_ibuildcube.
     */
    public abstract BDD buildCube(int value, int width, int[] var);
    
    /**
     * Builds a BDD variable set from an integer array.  The integer array v
     * holds the variable numbers.  The BDD variable set is represented by a
     * conjunction of all the variables in their positive form.
     * 
     * Compare to bdd_makeset.
     */
    public abstract void makeSet(int[] v);
    
    // TODO: bdd_scanset

    
    /**** STARTUP / SHUTDOWN ****/
    
    /**
     * Compare to bdd_init.
     * 
     * @param nodenum the initial number of BDD nodes
     * @param cachesize the size of caches used by the BDD operators
     */
    public abstract void initialize(int nodenum, int cachesize);

    /**
     * Compare to bdd_isrunning.
     * 
     * @return boolean
     */
    public abstract boolean isInitialized();

    /**
     * Resets the BDD package.  This function frees all memory used by the BDD
     * package and resets the package to its initial state.
     * 
     * Compare to bdd_done.
     */
    protected abstract void done();

    
    
    /**** CACHE/TABLE PARAMETERS ****/
    
    /**
     * Set the maximum available number of BDD nodes.
     * 
     * Compare to bdd_setmaxnodenum.
     */
    public abstract int setMaxNodeNum(int size);

    /**
     * Set minimum number of nodes to be reclaimed after a garbage collection.
     * The range of x is 0..100.  The default is 20.
     * 
     * Compare to bdd_setminfreenodes.
     * @param x
     */
    public abstract void setMinFreeNodes(int x);
    
    /**
     * Set maximum number of nodes used to increase node table.
     * 
     * Compare to bdd_setmaxincrease.
     * @param x
     */
    public abstract int setMaxIncrease(int x);
    
    /**
     * Sets the cache ratio for the operator caches.
     * 
     * Compare to bdd_setcacheratio.
     * @param x
     */
    public abstract int setCacheRatio(int x);
    
    
    
    /**** VARIABLE NUMBERS ****/
    
    /**
     * Returns the number of defined variables.
     * 
     * Compare to bdd_varnum.
     */
    public abstract int varNum();
    
    /**
     * Set the number of used BDD variables.  It can be called more than one
     * time, but only to increase the number of variables.
     * 
     * Compare to bdd_setvarnum.
     * 
     * @param x
     */
    public abstract void setVarNum(int num);
    
    /**
     * Add extra BDD variables.  Extends the current number of allocated BDD
     * variables with num extra variables.
     * 
     * Compare to bdd_extvarnum.
     * 
     * @param num
     */
    public abstract void extVarNum(int num);
    // TODO: handle error code for extvarnum
    
    /**
     * Returns a BDD representing the I'th variable.  (One node with the
     * children true and false).  The requested variable must be in the
     * (zero-indexed) range defined by setVarNum.
     * 
     * Compare to bdd_ithvar.
     * 
     * @param var the I'th variable on success, otherwise the constant false BDD
     */
    public abstract void ithVar(int var);
    
    /**
     * Returns a BDD representing the negation of the I'th variable.  (One node
     * with the children false and true).  The requested variable must be in the
     * (zero- indexed) range defined by setVarNum.
     * 
     * Compare to bdd_nithvar.
     * 
     * @param var the negated I'th variable on success, otherwise the constant
     * false BDD
     */
    public abstract void nithVar(int var);
    
    /**
     * Swap two BDD variables.
     */
    public abstract int swapVar(int v1, int v2);
    
    
    
    /**** INPUT / OUTPUT ****/
    
    /**
     * Prints all used entries in the node table.
     * 
     * Compare to bdd_printall.
     */
    public abstract void printAll();
    
    /**
     * Prints the description of a BDD in DOT format.
     * 
     * Compare to bdd_printdot.
     */
    public abstract void printDot();
    
    /**
     * Prints the node table entries used by a BDD.
     * 
     * Compare to bdd_printtable.
     */
    public abstract void printTable(BDD b);
    
    /**
     * Loads a BDD from a file.
     * 
     * Compare to bdd_load.
     */
    public abstract void load(String filename);
    // TODO: error code from bdd_load
    
    /**
     * Saves a BDD to a file.
     * 
     * Compare to bdd_save.
     */
    public abstract void save(String filename);
    // TODO: error code from bdd_save
    
    // TODO: bdd_strm_hook, bdd_file_hook, bdd_blockfile_hook
    // TODO: bdd_varprofile|
    // TODO: bdd_versionnum, bdd_versionstr
    
    
    
    /**
     * Compare to bdd_exist.
     */
    public abstract void exist();
    
    /**
     * Compare to bdd_restrict.
     */
    public abstract void restrict();
    
    
    
    /**
     * Compare to bdd_level2var.
     */
    public abstract int level2Var(int level);
    
    /**
     * Compare to bdd_var2level.
     */
    public abstract int var2level(int var);
    
    
    /**** REORDERING ****/
    
    /**
     * Compare to bdd_reorder.
     */
    public abstract void reorder(ReorderMethod m);
    
    /**
     * Enables automatic reordering.  If method is REORDER_NONE then automatic
     * reordering is disabled.
     * 
     * Compare to bdd_autoreorder.
     */
    public abstract void autoReorder(ReorderMethod method);
    
    /**
     * Enables automatic reordering with the given (maximum) number of
     * reorderings. If method is REORDER_NONE then automatic reordering is
     * disabled.
     * 
     * Compare to bdd_autoreorder_times.
     */
    public abstract void autoReorder(ReorderMethod method, int max);

    /**
     * Returns the current reorder method as defined by autoReorder.
     * 
     * Compare to bdd_getreorder_method.
     * 
     * @return ReorderMethod
     */
    public abstract ReorderMethod getReorderMethod();
    
    /**
     * Returns the number of allowed reorderings left.  This value can be
     * defined by autoReorder.
     * 
     * Compare to bdd_getreorder_times.
     */
    public abstract int getReorderTimes();
    
    /**
     * Disable automatic reordering until enableReorder is called.  Reordering
     * is enabled by default as soon as any variable blocks have been defined.
     * 
     * Compare to bdd_disable_reorder.
     */
    public abstract void disableReorder();
    
    /**
     * Enable automatic reordering after a call to disableReorder.
     * 
     * Compare to bdd_enable_reorder
     */
    public abstract void enableReorder();

    /**
     * Enables verbose information about reordering.  A value of zero means no
     * information, one means some information and greater than one means lots
     * of information.
     * 
     * @param v the new verbose level
     * @return the old verbose level
     */
    public abstract int reorderVerbose(int v);
    
    // TODO: bdd_setvarorder


    
    /**** VARIABLE BLOCKS ****/
    
    /**
     * Adds a new variable block for reordering.
     * 
     * Creates a new variable block with the variables in the variable set var.
     * The variables in var must be contiguous.
     * 
     * The fixed parameter sets the block to be fixed (no reordering of its
     * child blocks is allowed) or free,
     * 
     * Compare to bdd_addvarblock.
     */
    public abstract void addVarBlock(BDD var, boolean fixed);
    // TODO: handle error code for addVarBlock.
    
    /**
     * Adds a new variable block for reordering.
     * 
     * Creates a new variable block with the variables numbered first through
     * last, inclusive.
     * 
     * The fixed parameter sets the block to be fixed (no reordering of its
     * child blocks is allowed) or free,
     * 
     * Compare to bdd_intaddvarblock.
     */
    public abstract void addVarBlock(int first, int last, boolean fixed);
    // TODO: handle error code for addVarBlock.

    /**
     * Add a variable block for all variables.
     * 
     * Adds a variable block for all BDD variables declared so far.  Each block
     * contains one variable only.  More variable blocks can be added later with
     * the use of addVarBlock -- in this case the tree of variable blocks will
     * have the blocks of single variables as the leafs.
     * 
     * Compare to bdd_varblockall.
     */
    public abstract void varBlockAll();

    /**
     * Clears all the variable blocks that have been defined by calls to
     * addVarBlock.
     * 
     * Compare to bdd_clrvarblocks.
     */
    public abstract void clearVarBlocks();

    /**
     * Prints an indented list of the variable blocks.
     * 
     * Compare to bdd_printorder.
     */
    public abstract void printOrder();



    /**** BDD STATS ****/
    
    /**
     * Counts the number of shared nodes in a collection of BDDs.  Counts all
     * distinct nodes that are used in the BDDs -- if a node is used in more
     * than one BDD then it only counts once.
     * 
     * Compare to bdd_anodecount.
     */
    public abstract int nodeCount(Collection/*BDD*/ r);

    /**
     * Get the number of allocated nodes.  This includes both dead and active
     * nodes.
     * 
     * Compare to bdd_getallocnum.
     */
    public abstract int getAllocNum();

    /**
     * Get the number of active nodes in use.  Note that dead nodes that have
     * not been reclaimed yet by a garbage collection are counted as active.
     * 
     * Compare to bdd_getnodenum
     */
    public abstract int getNodeNum();

    /**
     * Calculate the gain in size after a reordering.  The value returned is
     * (100*(A-B))/A, where A is previous number of used nodes and B is current
     * number of used nodes.
     * 
     * Compare to bdd_reorder_gain.
     */
    public abstract int reorderGain();

    /**
     * Print cache statistics.
     * 
     * Compare to bdd_printstat.
     */
    public abstract void printStat();

    // TODO: bdd_cachestats, bdd_stats
    


    /**
     * Compare to bdd_swapvar.
     */
    public abstract void swapVar();
    
    /**
     * Compare to bdd_setvarorder.
     */
    public abstract void setVarOrder();
    
    // TODO: bdd_sizeprobe_hook, bdd_reorder_hook, bdd_resize_hook, bdd_gbc_hook
    // TODO: bdd_reorder_probe
    // TODO: bdd_error_hook, bdd_clear_error, bdd_errstring
    
    /**
     * Compare to fdd_extdomain.
     */
    public abstract void extDomain();
    
    // TODO: fdd_ithvar, fdd_equals, fdd_printset, fdd_setpair
    // TODO: fdd_ithset, fdd_makeset

    // TODO: bvec functions
    
    // TODO: bddPair, bdd_replace

    /**
     * @see java.lang.Object#finalize()
     */
    protected void finalize() throws Throwable {
        super.finalize();
        this.done();
    }

    /**
     * Reordering using a sliding window of 2.
     */
    public static ReorderMethod REORDER_WIN2    = new ReorderMethod(1, "WIN2");
    /**
     * Reordering using a sliding window of 2, iterating until no further
     * progress.
     */
    public static ReorderMethod REORDER_WIN2ITE = new ReorderMethod(2, "WIN2ITE");
    /**
     * Reordering using a sliding window of 3.
     */
    public static ReorderMethod REORDER_WIN3    = new ReorderMethod(3, "WIN3");
    /**
     * Reordering using a sliding window of 3, iterating until no further
     * progress.
     */
    public static ReorderMethod REORDER_WIN3ITE = new ReorderMethod(4, "WIN3ITE");
    /**
     * Reordering where each block is moved through all possible positions.  The
     * best of these is then used as the new position.  Potentially a very slow
     * but good method.
     */
    public static ReorderMethod REORDER_SIFT    = new ReorderMethod(5, "SIFT");
    /**
     * Same as REORDER_SIFT, but the process is repeated until no further
     * progress is done.  Can be extremely slow.
     */
    public static ReorderMethod REORDER_SIFTITE = new ReorderMethod(6, "SIFTITE");
    /**
     * Selects a random position for each variable.  Mostly used for debugging
     * purposes.
     */
    public static ReorderMethod REORDER_RANDOM  = new ReorderMethod(7, "RANDOM");
    
    public static class ReorderMethod {
        final int id; final String name;
        private ReorderMethod(int id, String name) {
            this.id = id;
            this.name = name;
        }
        public String toString() {
            return name;
        }
    }
}
