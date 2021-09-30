from jnius import JavaClass, MetaJavaClass, JavaMethod, JavaMultipleMethod, cast, autoclass

SizeBasedUniqueRandomXOR = autoclass(
    'org.spectrumauctions.sats.core.bidlang.xor.SizeBasedUniqueRandomXOR')
JavaUtilRNGSupplier = autoclass(
    'org.spectrumauctions.sats.core.util.random.JavaUtilRNGSupplier')
Random = autoclass('java.util.Random')
HashSet = autoclass('java.util.HashSet')
Bundle = autoclass(
    'org.marketdesignresearch.mechlib.core.Bundle')
BundleEntry = autoclass(
    'org.marketdesignresearch.mechlib.core.BundleEntry')

MRVM_MIP = autoclass(
    'org.spectrumauctions.sats.opt.model.mrvm.MRVM_MIP')

class _Mrvm(JavaClass, metaclass=MetaJavaClass):
    __javaclass__ = 'org/spectrumauctions/sats/core/model/mrvm/MultiRegionModel'

    #createNewPopulation = JavaMultipleMethod([
    #    '()Ljava/util/List;',
    #    '(J)Ljava/util/List;'])
    setNumberOfNationalBidders = JavaMethod('(I)V')
    setNumberOfRegionalBidders = JavaMethod('(I)V')
    setNumberOfLocalBidders = JavaMethod('(I)V')
    createWorld = JavaMethod(
        '(Lorg/spectrumauctions/sats/core/util/random/RNGSupplier;)Lorg/spectrumauctions/sats/core/model/mrvm/MRVMWorld;')
    createPopulation = JavaMethod(
        '(Lorg/spectrumauctions/sats/core/model/World;Lorg/spectrumauctions/sats/core/util/random/RNGSupplier;)Ljava/util/List;')

    def __init__(self, seed, number_of_national_bidders, number_of_regional_bidders, number_of_local_bidders):
        super().__init__()
        if seed:
            rng = JavaUtilRNGSupplier(seed)
        else:
            rng = JavaUtilRNGSupplier()

        self.population = {}
        self.goods = {}
        self.efficient_allocation = None
        self.setNumberOfNationalBidders(number_of_national_bidders)
        self.setNumberOfRegionalBidders(number_of_regional_bidders)
        self.setNumberOfLocalBidders(number_of_local_bidders)

        world = self.createWorld(rng)
        self._bidder_list = self.createPopulation(world, rng)

        # Store bidders
        bidderator = self._bidder_list.iterator()
        count = 0
        while bidderator.hasNext():
            bidder = bidderator.next()
            self.population[count] = bidder
            count += 1

        # Store goods
        goods_iterator = self._bidder_list.iterator().next().getWorld().getLicenses().iterator()
        while goods_iterator.hasNext():
            good = goods_iterator.next()
            self.goods[good.getLongId()] = good

        self.goods = list(map(lambda _id: self.goods[_id], sorted(self.goods.keys())))

    def get_model_name(self):
        return ('MRVM')

    def get_bidder_ids(self):
        return self.population.keys()

    # quick and dirty solution, needed the get_good_ids method for MRVM
    def get_good_ids(self):
        return dict.fromkeys(list(range(98))).keys()

    def calculate_value(self, bidder_id, goods_vector):
        assert len(goods_vector) == len(self.goods)
        bidder = self.population[bidder_id]
        bundleEntries = HashSet()
        for i in range(len(goods_vector)):
            if goods_vector[i] == 1:
                bundleEntries.add(BundleEntry(self.goods[i], 1))
        bundle = Bundle(bundleEntries)
        return bidder.calculateValue(bundle).doubleValue()

    def get_goods_of_interest(self, bidder_id):
        bidder = self.population[bidder_id]
        goods_of_interest = []
        for i in range(len(self.goods)):
            good_set = HashSet()
            good_set.add(self.goods[i])
            bundle = Bundle.of(good_set)
            if bidder.getValue(bundle, True).doubleValue() > 0:
                goods_of_interest.append(i)
        return goods_of_interest

    def get_uniform_random_bids(self, bidder_id, number_of_bids, seed=None):
        bidder = self.population[bidder_id]
        goods = autoclass('java.util.ArrayList')()
        for good in self.goods: goods.add(good)
        if seed:
            random = Random(seed)
        else:
            random = Random()

        bids = []
        for i in range(number_of_bids):
            bid = []
            bundle = bidder.getAllocationLimit().getUniformRandomBundle(random, goods)
            for i in range(len(self.goods)):
                if (bundle.contains(self.goods[i])):
                    bid.append(1)
                else:
                    bid.append(0)
            bid.append(bidder.getValue(bundle).doubleValue())
            bids.append(bid)
        return bids


    def get_random_bids(self, bidder_id, number_of_bids, seed=None, mean_bundle_size=49, standard_deviation_bundle_size=24.5):
        bidder = self.population[bidder_id]
        if seed:
            rng = JavaUtilRNGSupplier(seed)
        else:
            rng = JavaUtilRNGSupplier()
        valueFunction = cast('org.spectrumauctions.sats.core.bidlang.xor.SizeBasedUniqueRandomXOR',
                                bidder.getValueFunction(SizeBasedUniqueRandomXOR, rng))
        valueFunction.setDistribution(
            mean_bundle_size, standard_deviation_bundle_size)
        valueFunction.setIterations(number_of_bids)
        xorBidIterator = valueFunction.iterator()
        bids = []
        while (xorBidIterator.hasNext()):
            bundleValue = xorBidIterator.next()
            bid = []
            for i in range(len(self.goods)):
                if (bundleValue.getBundle().contains(self.goods[i])):
                    bid.append(1)
                else:
                    bid.append(0)
            bid.append(bundleValue.getAmount().doubleValue())
            bids.append(bid)
        return bids

    def get_efficient_allocation(self, display_output=True):
        """
        The efficient allocation is calculated on a generic definition. It is then "translated" into individual licenses that are assigned to bidders.
        Note that this does NOT result in a consistent allocation, since a single license can be assigned to multiple bidders.
        The value per bidder is still consistent, which is why this method can still be useful.
        """
        if self.efficient_allocation:
            return self.efficient_allocation, sum([self.efficient_allocation[bidder_id]['value'] for bidder_id in self.efficient_allocation.keys()])

        mip = MRVM_MIP(self._bidder_list)
        mip.setDisplayOutput(display_output)

        allocation = mip.calculateAllocation()

        self.efficient_allocation = {}

        for bidder_id, bidder in self.population.items():
            self.efficient_allocation[bidder_id] = {}
            self.efficient_allocation[bidder_id]['good_ids'] = []
            if allocation.getWinners().contains(bidder):
                bidder_allocation = allocation.allocationOf(bidder)
                bundle_entry_iterator = bidder_allocation.getBundle().getBundleEntries().iterator()
                while bundle_entry_iterator.hasNext():
                    bundle_entry = bundle_entry_iterator.next()
                    count = bundle_entry.getAmount()
                    licenses_iterator = cast('org.spectrumauctions.sats.core.model.mrvm.MRVMGenericDefinition', bundle_entry.getGood()).containedGoods().iterator()
                    for i in range(0, count):
                        assert licenses_iterator.hasNext()
                        self.efficient_allocation[bidder_id]['good_ids'].append(licenses_iterator.next().getLongId())

            self.efficient_allocation[bidder_id]['value'] = bidder_allocation.getValue().doubleValue() if allocation.getWinners().contains(bidder) else 0.0

        return self.efficient_allocation, allocation.getTotalAllocationValue().doubleValue()
