package Compil3r.Analysis.IPSSA;

import java.util.Set;

import Util.Assert;

import Clazz.jq_Method;
import Compil3r.Analysis.IPA.PA;
import Compil3r.Analysis.FlowInsensitive.MethodSummary.Node;
import Compil3r.Analysis.IPA.PAResults;
import Compil3r.Analysis.IPA.PointerAnalysisResults;
import Compil3r.Analysis.IPA.ProgramLocation.QuadProgramLocation;
import Compil3r.Quad.BasicBlock;
import Compil3r.Quad.CallGraph;

/**
 * This class returns pointer information in terms of ParametrizedLocation's.
 * 
 * @see ParametrizedLocation
 * */
class LocalPointerResults implements PointerAnalysisResults {
    PAResults _paResults;

    LocalPointerResults(PAResults paResults){
        this._paResults = paResults;
    }
    /** 
     * 
     * TODO: figure out a way to map from a Quad to the 
     * Node used there
     */
    public Set mod(QuadProgramLocation loc, BasicBlock bb) {
        if(IPSSABuilder.isStore(loc.getQuad())) {
            // Need to map from the Quad to a Node that is being stored to
            
        } else
        if(IPSSABuilder.isCall(loc.getQuad())) {
                       
        } 
        Assert._assert(false);
        return null;
    }
    public Set ref(QuadProgramLocation loc, BasicBlock block) {
        if(IPSSABuilder.isStore(loc.getQuad())) {
            // Need to map from the Quad to a Node that is being stored to
            
        } else
        if(IPSSABuilder.isCall(loc.getQuad())) {
                       
        } 
        Assert._assert(false);
        return null;
    }
    public Set getAliases(jq_Method method, SSALocation loc) {
        // TODO
        return null;
    }
    public boolean hasAliases(jq_Method method, SSALocation loc, ContextSet contextSet) {
        // TODO
        return false;
    }
    public boolean hasAliases(jq_Method method, SSALocation loc) {
        // TODO
        return false;
    }

    public PA getPAResults() {
        return _paResults.getPAResults();
    }

    /**
     * This is just taken from PAResults.
     * */
    public Set getCallTargets(QuadProgramLocation loc) {
        return _paResults.getCallTargets(loc);
    }
    public CallGraph getCallGraph() {
        return _paResults.getCallGraph();
    }
}

/**
 * This is a local SSALocation that is part of the Method summary.
 * @see LocalPointerProvider 
 * */
class ParametrizedLocation implements SSALocation {
    Node _node = null;
    
    ParametrizedLocation(Node node){
        this._node = node;
    }
    public String toString() {
        return _node.toString();
    } 
    public String toString(PA pa) {
        return toString();
    }
}