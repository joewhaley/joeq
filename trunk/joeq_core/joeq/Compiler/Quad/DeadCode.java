package Compil3r.Quad;

public class DeadCode extends Dataflow.EmptyAnalysis {
    static class TraceFact implements Dataflow.Fact {
        boolean _val;
	public TraceFact(boolean t) { _val = t; }

	public Dataflow.Fact deepCopy() { return new TraceFact(_val); }
	public Dataflow.Fact meetWith(Dataflow.Fact f) {
	    TraceFact other = (TraceFact) f;
	    _val = _val || other._val;
	    return this;
	}
	public boolean equals(Object o) {
	    if (o instanceof TraceFact) {
		return ((TraceFact) o)._val == _val;
	    }
	    return false;
	}
    }

    public void preprocess(ControlFlowGraph cfg) {
	_fc.setInitial(new TraceFact(true));
	_fc.setFinal(new TraceFact(false));
	QuadIterator qi = new QuadIterator(cfg);
	while (qi.hasNext()) {
	    Quad q = qi.nextQuad();
	    _fc.setPre(q, new TraceFact(false));
	    _fc.setPost(q, new TraceFact(false));
	}
    }

    public boolean transfer(Quad q) {
	Dataflow.Fact older = _fc.getPost(q).deepCopy();
	Dataflow.Fact newer = _fc.getPre(q).deepCopy();
	_fc.setPost(q, newer);
	return !newer.equals(older);
    }

    public void postprocess(ControlFlowGraph cfg) {
	QuadIterator qi = new QuadIterator(cfg);
	System.out.println("Results:");
	int deadCount = 0;
	while (qi.hasNext()) {
	    Quad q = qi.nextQuad();
	    if (((TraceFact)(_fc.getPre(q)))._val) continue;
	    ++deadCount;
	    System.out.println("UNREACHABLE: "+q);
	}
	if (deadCount == 0) {
	    System.out.println("All quads are reachable.");
	} else if (deadCount == 1) {
	    System.out.println("1 quad is unreachable.");
	} else {
	    System.out.println(deadCount + " quads are unreachable.");
	}	
    }
}
