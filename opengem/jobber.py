# -*- coding: utf-8 -*-
"""
Main jobber module
"""


from opengem import job
from opengem import flags
from opengem import hazard
from opengem import logs
from opengem import kvs
from opengem import producer
from opengem import risk
from opengem import shapes
from opengem import settings

from opengem.risk import tasks

from opengem.output import geotiff
from opengem.output.risk import RiskXMLWriter

from opengem.parser import exposure
from opengem.parser import hazard
from opengem.parser import vulnerability


FLAGS = flags.FLAGS

LOGGER = logs.LOG

class Jobber(object):
    """The Jobber class is responsible to evaluate the configuration settings
    and to execute the computations in parallel tasks (using the celery
    framework and the message queue RabbitMQ).
    """

    def __init__(self, job, partition):
        self.memcache_client = None
        self.partition = partition
        self.job = job
        
        self._init()
        self.block_id_generator = kvs.block_id_generator()

    def run(self):
        """Core method of Jobber. It splits the requested computation
        in blocks and executes these as parallel tasks.
        """

        job_id = self.job.id
        LOGGER.debug("running jobber, job_id = %s" % job_id)

        if self.partition is True:
            self._partition(job_id)
        else:
            block_id = self.block_id_generator.next()
            self._preload(job_id, block_id)
            self._execute(job_id, block_id)
            self._write_output_for_block(job_id, block_id)

        LOGGER.debug("Jobber run ended")

    def _partition(self, job_id):
        """
         _partition() has to:
          - get the full set of sites
          - select a subset of these sites
          - write the subset of sites to memcache, prepare a computation block
        """
        pass

    def _execute(self, job_id, block_id):
        """ Execute celery task for risk given block with sites """
        
        LOGGER.debug("starting task block, block_id = %s" % block_id)

        # task compute_risk has return value 'True' (writes its results to
        # memcache).
        result = tasks.compute_risk.apply_async(args=[job_id, block_id])

        # TODO(fab): Wait until result has been computed. This has to be
        # changed if we run more tasks in parallel.
        result.get()

    def _write_output_for_block(self, job_id, block_id):
        """note: this is usable only for one block"""
        
        # produce output for one block
        loss_curves = []

        sites = kvs.get_sites_from_memcache(self.memcache_client, job_id, 
            block_id)

        for (gridpoint, (site_lon, site_lat)) in sites:
            key = kvs.generate_product_key(job_id, 
                risk.LOSS_CURVE_KEY_TOKEN, block_id, gridpoint)
            loss_curve = self.memcache_client.get(key)
            loss_curves.append((shapes.Site(site_lon, site_lat), 
                                loss_curve))

        LOGGER.debug("serializing loss_curves")
        output_generator = RiskXMLWriter(settings.LOSS_CURVES_OUTPUT_FILE)
        output_generator.serialize(loss_curves)
        
        #output_generator = output.SimpleOutput()
        #output_generator.serialize(ratio_results)
        
        #output_generator = geotiff.GeoTiffFile(output_file, 
        #    region_constraint.grid)
        #output_generator.serialize(losses_one_perc)

    def _init(self):
        """ Initialize memcached_client. This should move into a Singleton """
        
        # TODO(fab): find out why this works only with binary=False
        self.memcache_client = kvs.get_client(binary=False)
        self.memcache_client.flush_all()

    def _preload(self, job_id, block_id):
        """ preload configuration for job """

        # set region
        # If there's a region file, use it. Otherwise,
        # get the region of interest as the convex hull of the
        # multipoint collection of the portfolio of assets.
        
        region_constraint = shapes.RegionConstraint.from_file(
                self.job[job.INPUT_REGION])

        # TODO(fab): the cell size has to be determined from the configuration 
        region_constraint.cell_size = 1.0

        # load hazard curve file and write to memcache_client
        nrml_parser = hazard.NrmlFile(self.job[job.HAZARD_CURVES])
        attribute_constraint = \
            producer.AttributeConstraint({'IMT' : 'MMI'})

        sites_hash_list = []

        for site, hazard_curve_data in nrml_parser.filter(
                region_constraint, attribute_constraint):

            gridpoint = region_constraint.grid.point_at(site)

            # store site hashes in memcache
            # TODO(fab): separate this from hazard curves. Regions of interest
            # should not be taken from hazard curve input, should be 
            # idependent from the inputs (hazard, exposure)
            sites_hash_list.append((str(gridpoint), 
                                   (site.longitude, site.latitude)))

            hazard_curve = shapes.Curve(zip(hazard_curve_data['IML'], 
                                                hazard_curve_data['Values']))

            memcache_key_hazard = kvs.generate_product_key(job_id, 
                hazard.HAZARD_CURVE_KEY_TOKEN, block_id, gridpoint)

            LOGGER.debug("Loading hazard curve %s at %s, %s" % (
                        hazard_curve, site.latitude,  site.longitude))

            success = self.memcache_client.set(memcache_key_hazard, 
                hazard_curve.to_json())

            if success is not True:
                raise ValueError(
                    "jobber: cannot write hazard curve to memcache")

        # write site hashes to memcache (JSON)
        memcache_key_sites = kvs.generate_sites_key(job_id, block_id)

        success = kvs.set_value_json_encoded(self.memcache_client, 
                memcache_key_sites, sites_hash_list)
        if not success:
            raise ValueError(
                "jobber: cannot write sites to memcache")
        
        # load assets and write to memcache
        exposure_parser = exposure.ExposurePortfolioFile(self.job['exposure'])
        for site, asset in exposure_parser.filter(region_constraint):
            gridpoint = region_constraint.grid.point_at(site)

            memcache_key_asset = kvs.generate_product_key(
                job_id, risk.EXPOSURE_KEY_TOKEN, block_id, gridpoint)

            LOGGER.debug("Loading asset %s at %s, %s" % (asset,
                site.longitude,  site.latitude))

            success = kvs.set_value_json_encoded(self.memcache_client, 
                memcache_key_asset, asset)
            if not success:
                raise ValueError(
                    "jobber: cannot write asset to memcache")

        # load vulnerability and write to memcache
        vulnerability.load_vulnerability_model(job_id,
            self.vulnerability_model_file)



